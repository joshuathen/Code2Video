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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Topology studies global shape properties.",
            "If you don't tear, holes define objects.",
            "Equivalent shapes are fundamentally the same."
        ]
        self.setup_layout("Summary and Conclusion", lecture_lines)
        
        # --- Define shapes ---
        square = Square(color=BLUE, fill_opacity=0.5)
        circle = Circle(color=GREEN, fill_opacity=0.5)
        triangle = Triangle(color=RED, fill_opacity=0.5)
        mug = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mug.svg", color=WHITE)
        
        label = Text("Topologically Equivalent", font_size=24, color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(square, 'C6', scale_factor=0.6)
        self.play(FadeIn(square))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        self.play(Transform(square, circle))
        self.place_at_grid(mug, 'D6', scale_factor=0.5)
        self.play(FadeIn(mug))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED)
        self.play(Transform(square, triangle))
        self.place_at_grid(label, 'E6', scale_factor=0.5)
        self.play(Write(label))
        self.wait(2)
        
        self.play(FadeOut(self.lecture), FadeOut(self.title), FadeOut(square), FadeOut(label), FadeOut(mug))
        self.wait(1)
