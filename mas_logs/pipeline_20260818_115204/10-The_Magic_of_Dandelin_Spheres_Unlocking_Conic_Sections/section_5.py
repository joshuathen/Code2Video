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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Conclusion", [
            "Dandelin spheres connect 3D cones to 2D geometry.", 
            "They reveal the foci as the conic DNA.", 
            "Geometry and algebra unite through these spheres."
        ])
        
        # Load asset
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        self.place_at_grid(sphere_asset, 'C4', scale_factor=1.5)
        
        # Define shapes
        c1 = Ellipse(width=1.5, height=1.0, color=WHITE)
        # Manim does not have a native Hyperbola class; constructing via ParametricFunction
        c2 = ParametricFunction(lambda t: np.array([1.5 * np.cosh(t), 1.5 * np.sinh(t), 0]), t_range=[-2, 2], color=WHITE)
        c3 = ParametricFunction(lambda t: np.array([t, 0.5 * t**2, 0]), t_range=[-1.5, 1.5], color=WHITE)
        
        # Applying requested grid positions
        self.place_at_grid(c1, 'B2', scale_factor=0.6)
        self.place_at_grid(c2, 'D2', scale_factor=0.6)
        self.place_at_grid(c3, 'B5', scale_factor=0.6)
        
        foci_points = VGroup(Dot(color=GOLD), Dot(color=GOLD), Dot(color=GOLD))
        foci_points[0].move_to(c1.get_center())
        foci_points[1].move_to(c2.get_center())
        foci_points[2].move_to(c3.get_center())
        
        # Final Text
        final_text = Text("Spheres Reveal Conic DNA", font_size=30, color=GOLD)
        self.place_at_grid(final_text, 'E4', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.play(
            FadeIn(c1), FadeIn(c2), FadeIn(c3),
            FadeIn(sphere_asset),
            self.lecture[0].animate.set_color(BLUE)
        )

        # === Animation for Lecture Line 2 ===
        self.play(
            FadeIn(foci_points),
            self.lecture[1].animate.set_color(GOLD)
        )

        # === Animation for Lecture Line 3 ===
        self.play(
            FadeIn(final_text),
            self.lecture[2].animate.set_color(GREEN)
        )
        
        self.wait(2)
