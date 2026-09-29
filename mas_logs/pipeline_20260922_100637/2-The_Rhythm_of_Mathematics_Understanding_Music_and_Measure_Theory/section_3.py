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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Defining the Time Signature", [
            "Time signatures act like mathematical formulas.",
            "The top number counts beats per measure.",
            "The bottom note value defines the beat."
        ])
        
        # Initialize Equation 4/4
        time_sig = MathTex(r"{4 \over 4}", font_size=96)
        top_num = time_sig[0][0]
        bottom_num = time_sig[0][2]
        
        # Assets
        metronome = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg")
        clock = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg")

        # Fix 40: Position time signature and metronome
        self.place_in_area(time_sig, 'B3', 'B4', scale_factor=1.2)
        self.place_at_grid(metronome, 'B6', scale_factor=0.5)
        self.play(Write(time_sig), FadeIn(metronome))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        # Fix 28, 41: Label beats
        beats_label = Text("Beats per measure", font_size=24, color="#FF4500")
        self.place_at_grid(beats_label, 'C3', scale_factor=0.9)
        
        self.play(
            self.lecture[1].animate.set_color("#FF4500"),
            top_num.animate.set_color("#FF4500"),
            FadeIn(beats_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fix 28, 41: Label note type
        note_label = Text("Note type per beat", font_size=24, color="#00FF00")
        self.place_at_grid(note_label, 'D3', scale_factor=0.9)
        
        self.play(
            self.lecture[2].animate.set_color("#00FF00"),
            bottom_num.animate.set_color("#00FF00"),
            FadeIn(note_label)
        )
        self.wait(1)
        
        # Fix 29, 42: Pulse animation
        pulse = VGroup(*[Text(str(i), font_size=48, color="#FFFF00") for i in range(1, 5)])
        pulse.arrange(RIGHT, buff=0.5)
        self.place_at_grid(pulse, 'E4', scale_factor=1.0)
        
        # Add clock asset to first beat
        self.place_at_grid(clock, 'E2', scale_factor=0.5)
        
        for i in range(4):
            if i == 0:
                self.play(
                    Indicate(pulse[i], color="#FFD700", scale_factor=1.5),
                    FadeIn(clock)
                )
            else:
                self.play(Indicate(pulse[i], color="#FFD700", scale_factor=1.5))
