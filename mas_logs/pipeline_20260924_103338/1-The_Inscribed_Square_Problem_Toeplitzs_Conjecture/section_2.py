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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["We map point pairs to a surface.", "Points close together sit near the diagonal.", "This configuration space visualizes every possible pair."]
        self.setup_layout("Prerequisite: The Configuration Space", lecture_lines)
        
        # Load Assets
        # Grid/Coordinates/Plane/Midpoint
        # Using SVG for placeholders if actual files don't resolve, but following instructions:
        grid_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        coords_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coordinates.svg")
        plane_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
        midpoint_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/midpoint.svg")
        
        # Adjust layout based on feedback
        self.place_in_area(grid_svg, 'B2', 'E5', scale_factor=0.6)
        
        # Labels for axes
        label_x = Text("x", color="#00FFFF")
        label_y = Text("y", color="#00FFFF")
        self.place_at_grid(label_x, 'E5', scale_factor=0.8)
        self.place_at_grid(label_y, 'B2', scale_factor=0.8)

        # Curve representation
        curve = ParametricFunction(lambda t: np.array([np.cos(t), np.sin(t), 0]), t_range=[0, TAU], color="#FF00FF")
        self.place_in_area(curve, 'B2', 'E5', scale_factor=0.5)
        
        dot1 = Dot(color="#00FFFF")
        dot2 = Dot(color="#00FFFF")
        midpoint = Dot(color="#FF00FF")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(grid_svg), FadeIn(coords_svg), self.lecture[0].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(plane_svg), FadeIn(dot1), FadeIn(dot2), self.lecture[1].animate.set_color("#FF00FF"))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(midpoint_svg), FadeIn(midpoint), self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
