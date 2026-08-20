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
        self.setup_layout("The Concept of Basis", [
            "Basis is the minimal spanning set.",
            "Linearly independent vectors efficiently define space.",
            "Basis is the most efficient coordinate system."
        ])
        
        # Axes
        axes = Axes(x_range=[-1, 4], y_range=[-1, 3], axis_config={"include_tip": True})
        self.place_in_area(axes, "C2", "F6", scale_factor=0.6)
        
        i_hat = Vector(RIGHT, color="#FF5733")
        j_hat = Vector(UP, color="#33FF57")
        
        # Setup position vectors in scene
        i_hat.move_to(axes.c2p(0, 0))
        j_hat.move_to(axes.c2p(0, 0))
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.play(Create(axes), Create(i_hat), Create(j_hat))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        
        # Animation: Scale to (3,2)
        # Using axes coordinate to keep it accurate
        target_pos = axes.c2p(3, 2)
        
        i_hat_new = Vector(3 * RIGHT, color="#FF5733").move_to(axes.c2p(1.5, 0))
        j_hat_new = Vector(2 * UP, color="#33FF57").move_to(axes.c2p(3, 1))
        
        self.play(
            ReplacementTransform(i_hat, i_hat_new),
            ReplacementTransform(j_hat, j_hat_new),
            run_time=2
        )
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#33AAFF")
        # Load asset grid.svg
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_asset, "A2", "F5", scale_factor=0.9)
        self.play(FadeIn(grid_asset))
        
        self.wait(2)
