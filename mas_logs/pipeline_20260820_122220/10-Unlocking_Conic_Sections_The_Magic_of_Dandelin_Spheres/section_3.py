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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Two spheres define the ellipse's foci.",
            "Points of tangency mark the focal spots.",
            "Distance sums stay constant for any point.",
            "This defines the elegant ellipse structure.",
            "Geometry connects points to simple curves."
        ]
        self.setup_layout("The Geometry of the Ellipse", lecture_lines)
        self.lecture.set_opacity(0)
        
        # Assets
        sphere_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        
        # Setup objects
        ellipse = Ellipse(width=4, height=2.5, color="#FF4500")
        self.place_in_area(ellipse, "A2", "E5", scale_factor=0.6)
        
        sphere_icon = SVGMobject(sphere_asset).set_color(BLUE)
        self.place_at_grid(sphere_icon, "A5", scale_factor=0.5)

        focus_left = Dot(color=YELLOW)
        focus_right = Dot(color=YELLOW)
        self.place_at_grid(focus_left, "C3", scale_factor=0.4)
        self.place_at_grid(focus_right, "C4", scale_factor=0.4)
        foci = VGroup(focus_left, focus_right)
        
        p = ValueTracker(0)
        def get_ellipse_point():
            return ellipse.point_from_proportion(p.get_value())
            
        point_on_ellipse = Dot(color=WHITE)
        point_on_ellipse.add_updater(lambda m: m.move_to(get_ellipse_point()))
        
        line1 = Line(focus_left.get_center(), point_on_ellipse.get_center(), color=BLUE)
        line2 = Line(focus_right.get_center(), point_on_ellipse.get_center(), color=BLUE)
        
        line1.add_updater(lambda m: m.put_start_and_end_on(focus_left.get_center(), point_on_ellipse.get_center()))
        line2.add_updater(lambda m: m.put_start_and_end_on(focus_right.get_center(), point_on_ellipse.get_center()))
        
        distance_lines = VGroup(line1, line2)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.play(FadeIn(sphere_icon), Create(ellipse))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        self.play(FadeIn(foci))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        self.add(point_on_ellipse, distance_lines)
        self.play(p.animate.set_value(1), run_time=3, rate_func=linear)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_opacity(1)
        self.play(Indicate(ellipse))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_opacity(1)
        self.wait(1)
