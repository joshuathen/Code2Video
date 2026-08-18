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
        self.setup_layout("Scalar Multiplication: Stretching Space", [
            "Scalars stretch or shrink a vector's length.",
            "A negative scalar flips the direction.",
            "The line's slope remains constant throughout."
        ])
        
        # Setup Axes
        axes = Axes(x_range=[-4, 4, 1], y_range=[-4, 4, 1], axis_config={"include_tip": True})
        self.place_in_area(axes, "C3", "F6", scale_factor=0.6)
        self.add(axes)
        
        v = Vector([1, 2], color="#FFFF00")
        self.place_at_grid(v, "D5", scale_factor=0.8)
        
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        self.place_at_grid(ruler, "A2", scale_factor=0.5)
        
        magnet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnet.svg")
        self.place_at_grid(magnet, "A2", scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(Create(v), FadeIn(ruler))
        self.wait(1)
        
        v2 = Vector([2, 4], color="#FFFF00")
        self.place_at_grid(v2, "D5", scale_factor=0.8)
        self.play(Transform(v, v2), FadeOut(ruler))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"), FadeIn(magnet))
        v_neg = Vector([-1, -2], color="#FF00FF")
        self.place_at_grid(v_neg, "D5", scale_factor=0.8)
        self.play(Transform(v, v_neg))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"), FadeOut(magnet))
        line = Line(start=axes.c2p(-2, -4), end=axes.c2p(2, 4), color="#00FFFF")
        self.add(line)
        self.wait(2)
