from manim import *
import numpy as np

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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup Layout
        title_text = "Summary: The Geometry of Roughness"
        lecture_lines = [
            "Fractal dimension measures an object's complexity and roughness.",
            "It reveals how shapes fill space as we zoom.",
            "Mathematics describes the beautiful patterns of our world."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        C_ORANGE = "#FF8C00"
        C_CYAN = "#00FFFF"
        C_GOLD = "#FFD700"

        # === Animation for Lecture Line 1 ===
        # Fast-paced montage: Coastline, Sierpinski Triangle, and Branching Lung
        self.play(self.lecture[0].animate.set_color(C_ORANGE))
        
        # Montage 1: Coastline (Jagged line)
        coast_pts = [
            np.array([x, np.sin(x*5)*0.2 + np.cos(x*12)*0.15, 0]) 
            for x in np.linspace(-1.8, 1.8, 25)
        ]
        coastline = VMobject(color=C_ORANGE).set_points_as_corners(coast_pts)
        self.place_in_area(coastline, "B1", "E6")
        self.play(Create(coastline), run_time=0.8)
        self.wait(0.4)
        self.play(FadeOut(coastline), run_time=0.4)

        # Montage 2: Sierpinski Triangle (Basic representation)
        sierpinski = VGroup(
            Triangle(color=C_ORANGE).scale(1.5),
            Triangle(color=BLACK, fill_opacity=1).scale(0.75).rotate(PI)
        )
        self.place_in_area(sierpinski, "B1", "E6")
        self.play(FadeIn(sierpinski), run_time=0.8)
        self.wait(0.4)
        self.play(FadeOut(sierpinski), run_time=0.4)

        # Montage 3: Branching Lung pattern
        lung = VGroup()
        def build_tree(start, angle, length, depth):
            if depth == 0: return
            end = start + np.array([np.cos(angle), np.sin(angle), 0]) * length
            lung.add(Line(start, end, color=C_ORANGE, stroke_width=depth*0.8))
            build_tree(end, angle + 0.4, length * 0.72, depth - 1)
            build_tree(end, angle - 0.4, length * 0.72, depth - 1)
        
        build_tree(ORIGIN + DOWN*0.8, PI/2, 0.9, 5)
        self.place_in_area(lung, "B1", "E6")
        self.play(Create(lung), run_time=0.8)
        self.wait(0.4)
        self.play(FadeOut(lung), run_time=0.4)

        # === Animation for Lecture Line 2 ===
        # Zoom into a Koch curve indefinitely
        self.play(self.lecture[1].animate.set_color(C_CYAN))

        def get_koch_points(p1, p2, depth):
            if depth == 0:
                return [p1, p2]
            v = p2 - p1
            s1 = p1 + v / 3
            s3 = p1 + 2 * v / 3
            unit_v = v / np.linalg.norm(v)
            normal = np.array([-unit_v[1], unit_v[0], 0])
            s2 = (p1 + p2) / 2 + normal * (np.linalg.norm(v) * np.sqrt(3) / 6)
            
            return (get_koch_points(p1, s1, depth-1)[:-1] +
                    get_koch_points(s1, s2, depth-1)[:-1] +
                    get_koch_points(s2, s3, depth-1)[:-1] +
                    get_koch_points(s3, p2, depth-1))

        start_p, end_p = LEFT * 2, RIGHT * 2
        
        # Display progression of detail (Iteration 1 to 4)
        current_koch = VMobject(color=C_CYAN)
        current_koch.set_points_as_corners(get_koch_points(start_p, end_p, 1))
        self.place_in_area(current_koch, "B1", "E6")
        self.play(Create(current_koch))

        for d in range(2, 5):
            next_pts = get_koch_points(start_p, end_p, d)
            next_koch = VMobject(color=C_CYAN)
            next_koch.set_points_as_corners(next_pts)
            self.place_in_area(next_koch, "B1", "E6")
            
            # Simulate zooming in by scaling the curve up while showing new detail
            self.play(
                current_koch.animate.scale(1.5).set_stroke(opacity=0),
                FadeIn(next_koch),
                run_time=1.2
            )
            current_koch = next_koch

        self.wait(1)
        self.play(FadeOut(current_koch))

        # === Animation for Lecture Line 3 ===
        # Final text appears in gold
        self.play(self.lecture[2].animate.set_color(C_GOLD))

        final_text = Text("Fractal Geometry:\nThe Language of Nature", 
                          color=C_GOLD, font_size=32, weight=BOLD)
        self.place_in_area(final_text, "A1", "F6")
        
        self.play(Write(final_text))
        # Gentle scale effect for the finale
        self.play(final_text.animate.scale(1.1), rate_func=there_and_back, run_time=2.5)
        self.wait(3)
