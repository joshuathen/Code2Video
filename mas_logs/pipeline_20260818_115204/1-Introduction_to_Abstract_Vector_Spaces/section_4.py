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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Applications: The Zoo of Vectors", [
            "Vector spaces include polynomials and functions.",
            "This zoo shows the versatility of vectors.",
            "Engineers use these to analyze complex signals."
        ])
        
        # Load assets
        mic = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        scope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/oscilloscope.svg")
        
        # Polynomials
        poly = MathTex("f(x) = ax^2 + bx + c", color="#9370DB")
        poly_label = Text("Polynomials", font_size=24, color="#9370DB")
        poly_group = VGroup(poly, poly_label, mic).arrange(DOWN)
        self.place_in_area(poly_group, 'A4', 'B6', scale_factor=0.6)

        # Signal wave
        wave = FunctionGraph(lambda x: 0.5 * np.sin(3 * x), x_range=[-2, 2], color="#00FF7F")
        wave_label = Text("Signals", font_size=24, color="#00FF7F")
        wave_group = VGroup(wave, wave_label).arrange(DOWN)
        self.place_in_area(wave_group, 'D4', 'E6', scale_factor=0.6)

        # Zoo Label
        zoo_label = Text("Vector Zoo", font_size=36, color="#FF6347")
        zoo_group = VGroup(zoo_label, scope).arrange(DOWN)
        self.place_at_grid(zoo_group, 'C2', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(poly_group), self.lecture[0].animate.set_color("#9370DB"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(wave_group), self.lecture[1].animate.set_color("#00FF7F"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(zoo_group), self.lecture[2].animate.set_color("#FF6347"))
        self.wait(2)
