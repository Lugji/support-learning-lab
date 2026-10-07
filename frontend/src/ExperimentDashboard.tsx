import { useEffect, useState } from 'react'
import './ExperimentDashboard.css'

type Result = {
  strategy: 'random' | 'margin'
  labeled_examples: number
  mean: number
  std: number
}

type Experiment = {
  dataset: string
  seeds: number[]
  weighting: 'none' | 'balanced'
  results: Result[]
}

const strategies = [
  { id: 'random', name: 'Zufällige Auswahl', color: '#64748b' },
  { id: 'margin', name: 'Margin-Auswahl', color: '#15803d' },
] as const

export default function ExperimentDashboard() {
  const [weighting, setWeighting] = useState<'none' | 'balanced'>('balanced')
  const [data, setData] = useState<Experiment | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    setData(null)
    setError('')

    async function load() {
      try {
        const response = await fetch(
          `/api/experiments?weighting=${weighting}`,
          { signal: controller.signal },
        )
        if (!response.ok) {
          throw new Error(`Experimentdaten konnten nicht geladen werden (${response.status}).`)
        }
        const result: Experiment = await response.json()
        if (!controller.signal.aborted) setData(result)
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : 'Verbindungsfehler.')
        }
      }
    }

    void load()
    return () => controller.abort()
  }, [weighting])

  const budgets = data
    ? [...new Set(data.results.map(row => row.labeled_examples))].sort((a, b) => a - b)
    : []

  const rows = budgets.map(budget => ({
    budget,
    random: data?.results.find(row =>
      row.strategy === 'random' && row.labeled_examples === budget),
    margin: data?.results.find(row =>
      row.strategy === 'margin' && row.labeled_examples === budget),
  }))

  const last = rows[rows.length - 1]
  const difference = last?.random && last?.margin
    ? (last.margin.mean - last.random.mean) * 100
    : null

  const x = (budget: number) => {
    const min = budgets[0] ?? 0
    const max = budgets[budgets.length - 1] ?? min
    return 70 + ((budget - min) / (max - min || 1)) * 620
  }
  const y = (score: number) => 310 - score * 270

  return (
    <section className="experiment-dashboard" aria-labelledby="experiment-title">
      <p className="experiment-eyebrow">EXPERIMENT LAB</p>
      <h2 id="experiment-title">Welche Beispiele helfen dem Modell?</h2>
      <p>
        Wir vergleichen zufällige Auswahl mit Margin-Auswahl:
        Das Modell wählt Texte, bei denen seine zwei besten Kategorien
        besonders nah beieinanderliegen.
      </p>

      <label className="experiment-filter">
        Klassengewichtung
        <select
          value={weighting}
          onChange={event => {
            setData(null)
            setError('')
            setWeighting(event.target.value as 'none' | 'balanced')
          }}
        >
          <option value="balanced">Mit Gewichtung — balanced</option>
          <option value="none">Ohne Gewichtung</option>
        </select>
      </label>

      <div aria-live="polite">
        {error && <p role="alert">{error} Prüfe, ob das Backend läuft.</p>}
        {!data && !error && <p>Experimentdaten werden geladen …</p>}
      </div>

      {data && (
        <>
          <div className="experiment-stats">
            <div><span>Datensatz</span><strong>{data.dataset}</strong></div>
            <div><span>Zufallsstarts</span><strong>{data.seeds.length}</strong></div>
            <div>
              <span>Margin − Zufall bei {last?.budget} Labels</span>
              <strong>
                {difference === null ? '—' : `${difference > 0 ? '+' : ''}${difference.toFixed(1)} F1-Punkte`}
              </strong>
            </div>
          </div>

          <div className="experiment-legend">
            {strategies.map(strategy => (
              <span key={strategy.id}>
                <i style={{ background: strategy.color }} />
                {strategy.name}
              </span>
            ))}
          </div>

          <svg
            className="experiment-chart"
            viewBox="0 0 740 375"
            role="img"
            aria-labelledby="learning-curve-title learning-curve-description"
          >
            <title id="learning-curve-title">Active-Learning-Lernkurven</title>
            <desc id="learning-curve-description">
              Macro-F1 auf den Validierungsdaten nach Anzahl gelabelter Beispiele.
              Punkte zeigen Mittelwerte, vertikale Balken eine Standardabweichung.
              Die genauen Zahlen stehen in der folgenden Tabelle.
            </desc>

            {[0, 0.2, 0.4, 0.6, 0.8, 1].map(tick => (
              <g key={tick}>
                <line x1="70" x2="690" y1={y(tick)} y2={y(tick)} stroke="#e2e8f0" />
                <text x="55" y={y(tick) + 4} textAnchor="end">{tick.toFixed(1)}</text>
              </g>
            ))}

            {budgets.map(budget => (
              <text key={budget} x={x(budget)} y="334" textAnchor="middle">
                {budget}
              </text>
            ))}
            <text x="380" y="363" textAnchor="middle">Gelabelte Beispiele</text>
            <text x="70" y="22">Macro-F1 · Validierung</text>

            {strategies.map(strategy => {
              const points = data.results
                .filter(row => row.strategy === strategy.id)
                .sort((a, b) => a.labeled_examples - b.labeled_examples)

              return (
                <g key={strategy.id} stroke={strategy.color}>
                  <polyline
                    points={points.map(row => `${x(row.labeled_examples)},${y(row.mean)}`).join(' ')}
                    fill="none"
                    strokeWidth="3"
                    strokeDasharray={strategy.id === 'random' ? '7 5' : undefined}
                  />
                  {points.map(row => {
                    const px = x(row.labeled_examples)
                    const top = y(Math.min(1, row.mean + row.std))
                    const bottom = y(Math.max(0, row.mean - row.std))
                    return (
                      <g key={row.labeled_examples}>
                        <line x1={px} x2={px} y1={top} y2={bottom} />
                        <line x1={px - 4} x2={px + 4} y1={top} y2={top} />
                        <line x1={px - 4} x2={px + 4} y1={bottom} y2={bottom} />
                        <circle cx={px} cy={y(row.mean)} r="4" fill={strategy.color} />
                      </g>
                    )
                  })}
                </g>
              )
            })}
          </svg>

          <div className="experiment-table-wrap">
            <table>
              <caption>Macro-F1: Mittelwert ± Standardabweichung</caption>
              <thead>
                <tr><th scope="col">Labels</th><th scope="col">Zufall</th><th scope="col">Margin</th></tr>
              </thead>
              <tbody>
                {rows.map(row => (
                  <tr key={row.budget}>
                    <th scope="row">{row.budget}</th>
                    <td>{row.random ? `${row.random.mean.toFixed(3)} ± ${row.random.std.toFixed(3)}` : '—'}</td>
                    <td>{row.margin ? `${row.margin.mean.toFixed(3)} ± ${row.margin.std.toFixed(3)}` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="experiment-note">
            Vorläufiges Simulationsexperiment mit bekannten Datensatz-Labels.
            Start: fünf Beispiele je Kategorie. Gleiche Validierungsdaten und
            gleiche Startbeispiele pro Zufallsstart. Balken zeigen die Streuung
            über drei Starts, keine Konfidenzintervalle. Der offizielle Testdatensatz
            wurde für diese Experimente nicht verwendet.
          </p>
        </>
      )}
    </section>
  )
}
