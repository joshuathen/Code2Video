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
        self.setup_layout("Summary & Conceptual Application", ["The vector itself never changes.", "Only its coordinate description shifts.", "Key for graphics and PCA."])
        
        # Load asset
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # Setup visual elements
        axes_a = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True}).scale(0.5)
        axes_b = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True}).scale(0.5)
        vector = Arrow(ORIGIN, [1, 1, 0], color=YELLOW)
        
        group_a = VGroup(axes_a, vector.copy(), grid_asset.copy())
        group_b = VGroup(axes_b, vector.copy(), grid_asset.copy())
        
        # Initial positions
        self.place_in_area(group_a, 'C4', 'D5', scale_factor=0.6)
        self.place_in_area(group_b, 'E4', 'F5', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#D3D3D3"))
        self.play(Create(group_a), Create(group_b))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#D3D3D3"))
        self.play(group_b.animate.rotate(PI/4))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#D3D3D3"))
        # Scale grid of group_b relative to group_a
        self.play(group_b[2].animate.scale(1.2))
        self.wait(2)
