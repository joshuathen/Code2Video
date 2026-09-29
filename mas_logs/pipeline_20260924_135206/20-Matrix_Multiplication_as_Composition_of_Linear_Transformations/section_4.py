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
        self.setup_layout("Deriving the Rule: Why 'Row by Column'?", [
            "Matrix multiplication is not commutative, so AB does not equal BA.",
            "It tracks where basis vectors end up after two mappings.",
            "Scaling then flipping differs from flipping then scaling."
        ])
        
        # Representations for shape [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/shape.svg]
        shape_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shape.svg")
        shape_group = VGroup(shape_icon)
        self.place_in_area(shape_group, 'B4', 'C6', scale_factor=0.7)
        self.play(FadeIn(shape_group))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        # Visualize basis vectors
        axes = Axes(x_length=2, y_length=2).add_coordinates()
        self.place_at_grid(axes, 'D4', scale_factor=0.7)
        i_hat = Vector(RIGHT, color=RED)
        j_hat = Vector(UP, color=GOLD)
        basis = VGroup(i_hat, j_hat).move_to(axes.c2p(0,0))
        self.add(basis)
        self.play(FadeIn(axes), Create(basis))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        
        # Visual demonstration: Scaling then Flipping vs Flipping then Scaling
        self.play(shape_group.animate.scale(0.5).set_color(RED))
        self.play(shape_group.animate.flip(axis=UP))
        self.wait(1)
