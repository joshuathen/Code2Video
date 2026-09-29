from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Introduction: The Heartbeat of Music", [
            "Music is organized by steady pulses.",
            "We group these pulses to create structure.",
            "This fundamental unit of time is a measure."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display steady pulse with glowing #FFD700 circle.
        pulse = Circle(radius=0.5, color="#FFD700", stroke_width=4)
        pulse.add(Dot(color="#FFD700", radius=0.6))
        self.place_at_grid(pulse, 'D4', scale_factor=0.9)
        self.play(FadeIn(pulse))
        
        def pulse_updater(m, dt):
            m.scale(1 + 0.1 * np.sin(self.time * 2 * PI))
        pulse.add_updater(pulse_updater)
        self.lecture[0].set_color("#FFD700")
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Show waveform visual, emphasizing rhythmic regularity, #FFFFFF color.
        waveform = VGroup()
        for i in range(10):
            line = Line(start=UP*0.5, end=DOWN*0.5, color=WHITE)
            waveform.add(line)
        waveform.arrange(RIGHT, buff=0.2)
        self.place_in_area(waveform, 'E2', 'F4', scale_factor=0.6)
        self.play(FadeIn(waveform))
        self.lecture[1].set_color("#FFFFFF")
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Add metronome [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg] visual clicking on strong beats, #FF4500.
        metronome = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg")
        metronome.set_color("#FF4500")
        self.place_at_grid(metronome, 'C6', scale_factor=0.7)
        self.play(FadeIn(metronome))
        self.lecture[2].set_color("#FF4500")
        self.wait(2)
        
        # Cleanup updaters
        pulse.remove_updater(pulse_updater)
