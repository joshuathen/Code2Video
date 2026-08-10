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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Synthesis and Summary", [
            "Span defines your total reach.", 
            "Dependence marks redundant paths.", 
            "Basis builds the perfect frame."
        ])
        
        # Setup visualization objects
        # Use Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg
        triangle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg", color=WHITE)
        
        # We need specific points for the vertices to animate them
        # Extract 3 main points from the triangle's path for highlighting
        # Fixed: Concatenate standard Manim vectors to create a valid array of points
        pts = triangle.get_center() + np.array([LEFT.tolist(), DOWN.tolist(), [0, 0, 0]])
        span_dot = Dot(triangle.get_critical_point(LEFT), color=BLUE)
        dep_dot = Dot(triangle.get_critical_point(UP), color=RED)
        basis_dot = Dot(triangle.get_critical_point(RIGHT), color=GREEN)
        
        frame = VGroup(triangle, span_dot, dep_dot, basis_dot)

        # === Animation for Lecture Line 1 ===
        # Initial placement: B3 to E5, scale 0.6
        self.place_in_area(frame, 'B3', 'E5', scale_factor=0.6)
        self.play(FadeIn(frame), run_time=1)
        self.play(self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        # Re-placement: C2 to F4, scale 0.7
        self.play(frame.animate.move_to(self.grid['C3']), run_time=0.5) # Update area center
        self.play(Indicate(dep_dot), run_time=1)
        self.play(self.lecture[1].animate.set_color(RED))

        # === Animation for Lecture Line 3 ===
        # Re-placement: B4 to E6, scale 0.65
        self.play(frame.animate.move_to(self.grid['D5']), run_time=0.5) # Update area center
        self.play(Indicate(basis_dot), run_time=1)
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.play(FadeOut(frame), run_time=1)
