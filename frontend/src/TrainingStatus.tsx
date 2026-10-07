import { useEffect, useState } from 'react'

type Status = {
  model_version: string
  training: {
    training_examples: number
    human_annotations: number
    validation_examples: number
    macro_f1_before: number
    macro_f1_after: number
    difference: number
  } | null
}

const format = new Intl.NumberFormat('de-DE', {
  minimumFractionDigits: 4,
  maximumFractionDigits: 4,
})

export default function TrainingStatus() {
  const [status, setStatus] = useState<Status | null>(null)
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setError('')

    async function load() {
      try {
        const response = await fetch('/api/annotation/status', {
          signal: controller.signal,
        })
        if (!response.ok) {
          throw new Error('Der Modellstatus konnte nicht geladen werden.')
        }
        const result: Status = await response.json()
        if (!controller.signal.aborted) setStatus(result)
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : 'Verbindungsfehler.')
        }
      }
    }

    void load()
    return () => controller.abort()
  }, [refresh])

  const training = status?.training

  return (
    <section aria-labelledby="training-status-title">
      <h3 id="training-status-title">Aktives Annotation-Modell</h3>
      {error ? (
        <>
          <p role="alert">{error}</p>
          <button onClick={() => setRefresh(value => value + 1)}>
            Status erneut laden
          </button>
        </>
      ) : !status ? (
        <p role="status">Modellstatus wird geladen …</p>
      ) : (
        <>
          <p>Version: <code>{status.model_version}</code></p>
          {training ? (
            <>
              <dl>
                <dt>Trainingsbeispiele</dt>
                <dd>{training.training_examples}</dd>
                <dt>Davon manuell annotiert</dt>
                <dd>{training.human_annotations}</dd>
                <dt>Macro-F1 vor dem Training</dt>
                <dd>{format.format(training.macro_f1_before)}</dd>
                <dt>Macro-F1 nach dem Training</dt>
                <dd>{format.format(training.macro_f1_after)}</dd>
                <dt>Veränderung in F1-Prozentpunkten</dt>
                <dd>
                  {training.difference > 0 ? '+' : ''}
                  {format.format(training.difference * 100)}
                </dd>
              </dl>
              <p className="annotation-note">
                Gemessen auf {training.validation_examples} festen
                Validierungsbeispielen. Ein einzelner Lauf belegt noch
                keine allgemeine Verbesserung.
              </p>
            </>
          ) : (
            <p>Initiales Modell; noch kein weiterer Trainingslauf aktiviert.</p>
          )}
          <p className="annotation-note">
            Diese Version steuert die Annotation-Warteschlange.
            Die Textklassifikation oben verwendet das Baseline-Modell.
          </p>
        </>
      )}
    </section>
  )
}
