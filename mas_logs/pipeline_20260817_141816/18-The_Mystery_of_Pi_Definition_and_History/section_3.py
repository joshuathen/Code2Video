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
        lecture_lines = ["Ancient cultures estimated Pi differently.", "Archimedes used polygons to trap circles.", "More sides mean a better estimate."]
        self.setup_layout("Historical Evolution: From Estimation to Precision", lecture_lines)
        
        circle = Circle(radius=1.5, color=WHITE)
        self.place_at_grid(circle, 'C5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        # Show Egyptian/Babylonian approximations
        approx_text = VGroup(
            Text("Babylonians: π ≈ 3", font_size=24, color="#CCCCCC"),
            Text("Egyptians: π ≈ 3.16", font_size=24, color="#CCCCCC")
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(approx_text, 'E5', scale_factor=0.7)
        
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(approx_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Archimedes' polygon method
        polygon = RegularPolygon(n=4, radius=1.6, color=BLUE)
        self.place_at_grid(polygon, 'C5', scale_factor=0.6)
        
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Create(polygon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Increase sides
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        for n in [6, 8, 12, 24, 64]:
            new_polygon = RegularPolygon(n=n, radius=1.6, color=BLUE)
            self.place_at_grid(new_polygon, 'C5', scale_factor=0.6)
            self.play(Transform(polygon, new_polygon), run_time=0.5)
            
        modern_pi = Text("Modern π ≈ 3.14159", font_size=28, color="#00FFFF")
        self.place_at_grid(modern_pi, 'E6', scale_factor=0.7)
        self.play(Write(modern_pi))
        self.wait(2)
