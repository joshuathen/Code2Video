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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Divergence measures expansion or contraction at points.",
            "Positive divergence acts like a source.",
            "Negative divergence acts like a sink.",
            "Think of blowing air from a balloon.",
            "Or water swirling down a drain."
        ]
        self.setup_layout("Divergence: The Source or Sink", lecture_lines)
        
        # Elements
        div_text = Text("Divergence", color="#00ff00", font_size=40)
        self.place_at_grid(div_text, "B3", scale_factor=0.9)

        formula = MathTex(r"\nabla \cdot \mathbf{F}", color=WHITE, font_size=48)
        self.place_at_grid(formula, "E4", scale_factor=1.2)

        # Assets
        source_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/balloon.svg")
        sink_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drain.svg")
        
        source_label = Text("Source", color="#ff5555", font_size=20)
        sink_label = Text("Sink", color="#55ff55", font_size=20)

        self.place_at_grid(source_img, "D2", scale_factor=0.8)
        self.place_at_grid(source_label, "E2", scale_factor=0.7)
        
        self.place_at_grid(sink_img, "D5", scale_factor=0.8)
        self.place_at_grid(sink_label, "E5", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00ff00")
        self.play(FadeIn(div_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#ff5555")
        self.play(FadeIn(source_img), FadeIn(source_label), FadeIn(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#55ff55")
        self.play(FadeIn(sink_img), FadeIn(sink_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        self.play(source_img.animate.scale(1.2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        self.play(sink_img.animate.scale(0.8))
        self.wait(1)
