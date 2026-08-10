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
        lecture_lines = ["Basis is the minimal spanning set.", "Basis vectors must be linearly independent.", "Unit vectors (1,0) and (0,1) form a basis."]
        self.setup_layout("Bases: The Minimalist Toolkit", lecture_lines)
        
        # Assets
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True}).scale(0.5)
        vec1 = Vector([1, 0], color=YELLOW)
        vec2 = Vector([0, 1], color=BLUE)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg]
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        basis_group = VGroup(axes, vec1, vec2, grid_asset)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_in_area(basis_group, 'B2', 'D5', scale_factor=0.6)
        self.add(basis_group)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        checkmark = Tex(r"$\checkmark$", color=GREEN)
        self.place_at_grid(checkmark, 'D5', scale_factor=0.5)
        self.play(Write(checkmark))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        label1 = MathTex(r"\vec{i}=(1,0)", color=YELLOW)
        label2 = MathTex(r"\vec{j}=(0,1)", color=BLUE)
        
        self.place_at_grid(label1, 'D6', scale_factor=0.7)
        self.place_at_grid(label2, 'B4', scale_factor=0.7)
        
        self.play(Write(label1), Write(label2))
        self.wait(2)
