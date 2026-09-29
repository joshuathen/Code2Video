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
        self.setup_layout("Prerequisite Refresher: The Complex Plane", [
            "Complex points act as geometric transformations.",
            "Squaring maps rotate and scale points.",
            "Functions transform the entire complex plane."
        ])
        
        # Assets
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        ruler_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True, "color": WHITE})
        plane_group = VGroup(grid_asset, axes).scale(0.8)
        self.place_in_area(plane_group, 'C3', 'F6', scale_factor=0.6)
        
        z_val = 1 + 0.5j
        z = Dot(point=axes.c2p(z_val.real, z_val.imag), color=YELLOW)
        z_label = MathTex("z", color=YELLOW).next_to(z, UP)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(plane_group))
        self.play(Create(z), Write(z_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        z_squared = z_val**2
        z2 = Dot(point=axes.c2p(z_squared.real, z_squared.imag), color=BLUE)
        z2_label = MathTex("z^2", color=BLUE).next_to(z2, UP)
        
        # Ruler asset usage
        ruler = ruler_asset.copy()
        self.place_in_area(ruler, 'C4', 'D5', scale_factor=0.4)
        
        self.play(TransformFromCopy(z, z2), Write(z2_label), FadeIn(ruler))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.play(FadeOut(axes), FadeOut(z2_label), FadeOut(ruler))
        self.wait(2)
