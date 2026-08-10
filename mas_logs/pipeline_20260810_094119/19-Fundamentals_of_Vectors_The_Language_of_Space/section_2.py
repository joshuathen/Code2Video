from manim import *
import os

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
        self.setup_layout("Vector Representation and Components", [
            "We represent space using coordinates.",
            "A vector splits into horizontal, vertical parts.",
            "Moving three right, four up.",
            "Vector is [3, 4] from origin."
        ])
        
        # Coordinate system
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 6, 1], axis_config={"color": "#333333"})
        
        # Asset integration
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg"
        if os.path.exists(asset_path):
            grid_asset = SVGMobject(asset_path)
            self.place_in_area(grid_asset, 'C3', 'F6', scale_factor=0.55)
            self.add(grid_asset)
        
        # Applying requested layout: C3-F6 area, scale 0.55
        self.place_in_area(axes, 'C3', 'F6', scale_factor=0.55)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(axes), run_time=1)
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        vector = Arrow(axes.c2p(0, 0), axes.c2p(3, 4), color="#00CED1", buff=0)
        self.play(Create(vector), run_time=1)
        self.lecture[1].set_color(YELLOW)
        
        # === Animation for Lecture Line 3 ===
        horiz_line = DashedLine(axes.c2p(0, 0), axes.c2p(3, 0), color=WHITE)
        vert_line = DashedLine(axes.c2p(3, 0), axes.c2p(3, 4), color=WHITE)
        self.play(Create(horiz_line), Create(vert_line), run_time=1)
        self.lecture[2].set_color(YELLOW)
        
        # === Animation for Lecture Line 4 ===
        # Applying requested grid: B4, scale 0.7
        label = MathTex(r"\\vec{v} = [3, 4]", color=WHITE)
        self.place_at_grid(label, 'B4', scale_factor=0.7)
        self.play(Write(label), run_time=1)
        self.lecture[3].set_color(YELLOW)
        self.wait(1)
