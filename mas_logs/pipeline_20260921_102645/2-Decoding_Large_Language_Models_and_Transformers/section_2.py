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
        self.setup_layout("The Transformer Core: Attention Mechanism", [
            "Attention allows models to read words simultaneously.",
            "It assigns importance weights to every word.",
            "Context clarifies relationships between distant words.",
            "The model focuses on relevant connections.",
            "Complex sentence structure becomes understandable."
        ])
        
        # Assets
        book = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        brain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg")

        words = VGroup(
            Text("The", font_size=24),
            Text("cat", font_size=24),
            Text("sat", font_size=24)
        ).arrange(RIGHT, buff=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(book, 'A1', scale_factor=0.3)
        self.play(FadeIn(words), FadeIn(book))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        lines = VGroup(
            Line(words[0].get_bottom(), words[1].get_bottom(), stroke_width=4, color="#FFFF00"),
            Line(words[1].get_bottom(), words[2].get_bottom(), stroke_width=2, color="#FFFF00")
        )
        self.play(Create(lines))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        self.play(lines[0].animate.set_color("#FF0000").set_stroke(width=6))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        self.play(
            lines[1].animate.set_color("#00FF00").set_stroke(width=6),
            lines[0].animate.set_color("#FFFFFF").set_stroke(width=2)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.play(FadeOut(words), FadeOut(lines), FadeOut(book))
        self.place_at_grid(brain, 'C3', scale_factor=0.5)
        self.play(FadeIn(brain))
        self.wait(2)
