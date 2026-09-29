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
        self.setup_layout("Real-World Applications: From GPS to Image Processing", [
            "These curves optimize spatial database indexing.",
            "They also improve memory cache locality.",
            "Fractal antennas fit into tiny devices.",
            "They maintain excellent signal strength throughout.",
            "Spatial mapping is everywhere in technology."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show a map with points
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg", color=WHITE)
        dots = VGroup(*[Dot(radius=0.05, color=WHITE) for _ in range(10)])
        visual_group = VGroup(map_icon, dots)
        self.place_at_grid(visual_group, "B3", scale_factor=0.5)
        self.play(FadeIn(visual_group))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Use Hilbert curve to index locations
        self.lecture[1].set_color("#FF0000")
        curve = VMobject(color="#FF0000")
        curve.set_points_smoothly([self.grid["A2"], self.grid["A5"], self.grid["F5"], self.grid["F2"]])
        self.play(Create(curve))
        
        # === Animation for Lecture Line 3 ===
        # Demonstrate rapid lookup in indexed space
        self.lecture[2].set_color("#00FF00")
        header_text = Text("Grid Visual", font_size=20, color=BLUE)
        self.place_at_grid(header_text, "A1", scale_factor=0.9)
        
        dot_grid = VGroup(*[Dot(radius=0.04, color=BLUE) for _ in range(16)])
        dot_grid.arrange_in_grid(rows=4, cols=4, buff=0.2)
        self.place_in_area(dot_grid, "B1", "F6", scale_factor=0.7)
        
        highlight = Circle(radius=0.3, color="#00FF00")
        highlight.move_to(dot_grid[0])
        
        visual_group_3 = VGroup(header_text, dot_grid, highlight)
        self.place_in_area(visual_group_3, "C2", "F6", scale_factor=0.6)
        
        self.play(FadeIn(header_text), Create(dot_grid), FadeIn(highlight))
        self.play(highlight.animate.move_to(dot_grid[5]))
        self.wait(1)
