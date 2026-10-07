import ExperimentDashboard from './ExperimentDashboard'
import { useState } from 'react'
import './App.css'

type Prediction = {
  category: string
  probability: number
}

export default function App() {
  const [text, setText] = useState('My card has not arrived yet.')
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function analyze() {
    if (!text.trim() || loading) return

    setLoading(true)
    setError('')
    setPredictions([])

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      })

      if (!response.ok) {
        throw new Error(`Anfrage fehlgeschlagen (${response.status}).`)
      }

      const data = await response.json()
      setPredictions(data.predictions)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Anfrage fehlgeschlagen.',
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="lab">
      <header>
        <p className="eyebrow">SUPPORT LEARNING LAB</p>
        <h1>Supportanfragen verstehen.</h1>
        <p>
          Gib eine englische Bankanfrage ein und vergleiche die
          vorgeschlagenen Kategorien.
        </p>
      </header>

      <section className="panel">
        <label htmlFor="request">Deine Supportanfrage</label>

        <textarea
          id="request"
          rows={5}
          maxLength={5000}
          value={text}
          disabled={loading}
          onChange={(event) => {
            setText(event.target.value)
            setPredictions([])
            setError('')
          }}
        />

        <button
          onClick={analyze}
          disabled={loading || !text.trim()}
        >
          {loading ? 'Wird analysiert …' : 'Analysieren'}
        </button>

        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
      </section>

      <section
        className="panel"
        aria-live="polite"
        aria-busy={loading}
      >
        <h2>Top-3-Kategorien</h2>

        {predictions.length === 0 ? (
          <p>Die Ergebnisse erscheinen nach der Analyse.</p>
        ) : (
          <ol className="results">
            {predictions.map((prediction) => (
              <li key={prediction.category}>
                <span>
                  {prediction.category.replaceAll('_', ' ')}
                </span>
                <strong>
                  {(prediction.probability * 100).toFixed(1)} %
                </strong>
              </li>
            ))}
          </ol>
        )}

        <p className="note">
          Die Werte sind Modellschätzungen und keine Garantie
          für eine richtige Zuordnung.
        </p>
      </section>
      <ExperimentDashboard />
    </main>
  )
}
