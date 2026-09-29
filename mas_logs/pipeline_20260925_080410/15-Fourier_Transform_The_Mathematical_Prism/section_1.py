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
        lecture_lines = ["Light travels as waves, hidden together.", 
                         "A prism separates light into its colors.", 
                         "Fourier Transform is a mathematical prism.", 
                         "It splits complex signals into frequencies.", 
                         "Think of piano chords becoming individual notes."]
        self.setup_layout("Fourier Transform: The Mathematical Prism", lecture_lines)
        
        # Define elements
        white_beam = Line(start=LEFT*2, end=RIGHT*2, color=WHITE, stroke_width=4)
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        rainbow = VGroup(*[Line(ORIGIN, RIGHT*2, color=c, stroke_width=4) for c in ["#FF0000", "#FFA500", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#8A2BE2"]])
        rainbow.arrange(DOWN, buff=0.1)
        
        time_label = Text("Time Domain", color=WHITE, font_size=20)
        freq_label = Text("Frequency Domain", color="#00FFFF", font_size=20)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.play(Create(white_beam))
        self.place_at_grid(time_label, 'B3', scale_factor=0.7)
        self.play(FadeIn(time_label))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        self.place_at_grid(prism, 'C3', scale_factor=0.9)
        self.play(FadeIn(prism))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.place_at_grid(freq_label, 'C4', scale_factor=0.7)
        self.play(FadeIn(freq_label))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        self.place_in_area(rainbow, 'D4', 'E6', scale_factor=0.6)
        self.play(Transform(white_beam.copy(), rainbow), run_time=2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        glow = prism.copy().set_stroke(opacity=0.5, width=10)
        self.play(ShowPassingFlash(glow))
