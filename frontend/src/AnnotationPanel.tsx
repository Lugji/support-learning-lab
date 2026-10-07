import { useEffect, useState } from 'react'
import './AnnotationPanel.css'

type QueueData = {
  sample: {
    id: number
    text: string
    margin: number
    predictions: { category: string; probability: number }[]
  } | null
  categories: string[]
  completed: number
  remaining: number
}

export default function AnnotationPanel() {
  const [data, setData] = useState<QueueData | null>(null)
  const [category, setCategory] = useState('')
  const [refresh, setRefresh] = useState(0)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    setData(null)
    setCategory('')
    setError('')

    async function load() {
      try {
        const response = await fetch('/api/annotation/next', {
          signal: controller.signal,
        })
        if (!response.ok) throw new Error('Die Warteschlange konnte nicht geladen werden.')
        const result: QueueData = await response.json()
        if (!controller.signal.aborted) setData(result)
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : 'Verbindungsfehler.')
        }
      }
    }

    void load()
    return () => controller.abort()
  }, [refresh])

  async function save() {
    if (!data?.sample || !category || saving) return
    setSaving(true)
    setError('')
    setMessage('')

    try {
      const response = await fetch('/api/annotation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample_id: data.sample.id,
          category,
        }),
      })

      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(
          typeof body?.detail === 'string'
            ? body.detail
            : `Speichern fehlgeschlagen (${response.status}).`,
        )
      }

      setMessage('Deine Kategorie wurde gespeichert.')
      setData(null)
      setCategory('')
      setRefresh(value => value + 1)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Verbindungsfehler.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="annotation-panel" aria-labelledby="annotation-title">
      <p className="annotation-eyebrow">HUMAN IN THE LOOP</p>
      <h2 id="annotation-title">Hilf dem Modell mit deiner Entscheidung</h2>
      <p>
        Die Warteschlange beginnt mit Texten, bei denen das Modell
        zwischen seinen zwei besten Kategorien besonders unsicher ist.
      </p>

      <div aria-live="polite">
        {message && <p className="annotation-success">{message}</p>}
        {error && <p role="alert" className="annotation-error">{error}</p>}
        {!data && !error && <p>Nächster Text wird geladen …</p>}
      </div>

      {data && (
        <>
          <p className="annotation-progress">
            {data.completed} annotiert · {data.remaining} offen
          </p>

          {data.sample ? (
            <>
              <blockquote className="annotation-text">
                {data.sample.text}
              </blockquote>

              <label className="annotation-label" htmlFor="annotation-category">
                Welche Kategorie passt zu diesem Text?
              </label>
              <select
                id="annotation-category"
                value={category}
                disabled={saving}
                onChange={event => setCategory(event.target.value)}
              >
                <option value="">Kategorie auswählen …</option>
                {data.categories.map(item => (
                  <option key={item} value={item}>
                    {item.replaceAll('_', ' ')}
                  </option>
                ))}
              </select>

              <button
                className="annotation-save"
                disabled={!category || saving}
                onClick={() => void save()}
              >
                {saving ? 'Wird gespeichert …' : 'Speichern & nächsten Text laden'}
              </button>

              <details key={data.sample.id} className="annotation-suggestions">
                <summary>Modellvorschläge ansehen</summary>
                <p>Modellwerte sind keine garantierten Trefferwahrscheinlichkeiten.</p>
                <ol>
                  {data.sample.predictions.map(prediction => (
                    <li key={prediction.category}>
                      {prediction.category.replaceAll('_', ' ')}
                      {' · '}
                      {(prediction.probability * 100).toFixed(1)} %
                    </li>
                  ))}
                </ol>
              </details>
            </>
          ) : (
            <p>Alle Texte in dieser Warteschlange sind annotiert.</p>
          )}
        </>
      )}

      {error && (
        <button
          className="annotation-retry"
          disabled={saving}
          onClick={() => setRefresh(value => value + 1)}
        >
          Warteschlange neu laden
        </button>
      )}

      <p className="annotation-note">
        Deine Entscheidungen werden dauerhaft lokal gespeichert.
        Das Modell und die Reihenfolge bleiben bis zum nächsten Training
        auf ihrem bisherigen Stand.
      </p>
    </section>
  )
}
