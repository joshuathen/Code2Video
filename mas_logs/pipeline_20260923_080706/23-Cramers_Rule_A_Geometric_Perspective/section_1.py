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
        lecture_lines = [
            "Determinants represent areas of parallelograms.",
            "Vectors define the parallelogram's sides.",
            "Area equals the absolute determinant value."
        ]
        self.setup_layout("Prerequisites: Vectors as Areas", lecture_lines)
        
        # Axes and vectors
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": True})
        self.place_in_area(axes, "A1", "E4", scale_factor=0.5)
        
        v1 = Vector([2, 1], color=BLUE)
        v2 = Vector([1, 2], color=GREEN)
        
        # Parallelogram using grid-based coordinate mapping
        # Vectors (2, 1) and (1, 2)
        parallelogram = Polygon(ORIGIN, [2, 1, 0], [3, 3, 0], [1, 2, 0], color=YELLOW, fill_opacity=0.3)
        self.place_in_area(parallelogram, "B2", "E4", scale_factor=0.4)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        try:
            asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        except:
            asset_icon = Dot(color=RED) # Placeholder
            
        self.place_at_grid(asset_icon, "F4", scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Create(axes))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        # Position the vectors relative to the axes area
        v1.next_to(axes.get_origin(), RIGHT, buff=0)
        v2.next_to(axes.get_origin(), UP, buff=0)
        self.play(GrowArrow(v1), GrowArrow(v2))
        self.play(Create(parallelogram))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        area_text = MathTex(r"\\text{Area} = |det| = 3", color=YELLOW)
        self.place_at_grid(area_text, "F3", scale_factor=0.7)
        self.play(Write(area_text), FadeIn(asset_icon))
        self.wait(2)
