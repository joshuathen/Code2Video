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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup initial layout
        title_str = "The Map of All Maps: The Mandelbrot Set"
        lines = [
            "Each value of c creates a unique Julia set.",
            "The Mandelbrot set maps these different c values.",
            "If an orbit stays bounded, we color it black.",
            "Moving across the map morphs the Julia set's shape.",
            "It is the ultimate dictionary of complex dynamics."
        ]
        self.setup_layout(title_str, lines)

        # === Animation for Lecture Line 1 ===
        # "Each value of c creates a unique Julia set."
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Pop up several miniature Julia Set icons (#FFFFFF) in a circle around the center of the screen area.
        julia_icons = VGroup()
        # center of the visual area roughly at D4
        center_grid = self.grid["D4"]
        for i in range(6):
            # Using simple varied shapes to represent unique Julia sets
            icon = Star(n=5 + i, color="#FFFFFF", fill_opacity=0.3).scale(0.2)
            angle = i * 60 * DEGREES
            dist = 1.2
            icon_pos = center_grid + np.array([np.cos(angle)*dist, np.sin(angle)*dist, 0])
            icon.move_to(icon_pos)
            julia_icons.add(icon)
        
        self.play(LaggedStart(*[FadeIn(icon, scale=0.5) for icon in julia_icons], lag_ratio=0.1))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # "The Mandelbrot set maps these different c values."
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        
        # Mandelbrot silhouette: Cardioid + Main Bulb approximation
        mandel_cardioid = ParametricFunction(
            lambda t: np.array([
                0.25 + 0.5 * np.cos(t) - 0.25 * np.cos(2*t),
                0.5 * np.sin(t) - 0.25 * np.sin(2*t),
                0
            ]),
            t_range=[0, TAU],
            color="#FFFFFF"
        )
        mandel_bulb = Circle(radius=0.25, color="#FFFFFF").move_to(np.array([-0.75, 0, 0]))
        mandel_set = VGroup(mandel_cardioid, mandel_bulb)
        
        # [RESOLVED Issue 31]: Move to C3-F6 to avoid crowding and improve vertical utilization
        self.place_in_area(mandel_set, "C3", "F6", scale_factor=1.2)
        
        self.play(
            FadeOut(julia_icons),
            Create(mandel_set)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # "If an orbit stays bounded, we color it black."
        self.play(self.lecture[2].animate.set_color("#888888"))
        
        mandel_fill = mandel_set.copy().set_fill("#000000", opacity=1.0).set_stroke(width=1)
        self.play(FadeIn(mandel_fill))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # "Moving across the map morphs the Julia set's shape."
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        
        c_point = Dot(color="#FF0000").scale(1.2)
        # Starting point for c relative to mandel_set center
        c_point.move_to(mandel_set.get_center() + LEFT * 0.4)
        
        # Side-window for Julia Set morphing in area A5-B6
        side_window = Square(color="#FFFFFF", stroke_width=2).scale(0.8)
        # [RESOLVED Issue 33]: Ensure vertical utilization and correct placement
        self.place_in_area(side_window, "A5", "B6", scale_factor=0.9)
        julia_label = Text("Julia Set Morph", font_size=14, color="#FFFFFF").next_to(side_window, UP, buff=0.1)
        
        # Simple morphing shape using a RegularPolygon
        morph_shape = RegularPolygon(n=6, color="#FFFFFF", fill_opacity=0.2).move_to(side_window.get_center()).scale(0.5)
        
        # Updater for visual morphing effect
        def update_morph(mob):
            p = c_point.get_center()
            center = mandel_set.get_center()
            dist = np.linalg.norm(p - center)
            # Use stretch_to_fit_height to avoid aspect ratio lock if desired (L025)
            new_height = 0.6 + 0.3 * np.sin(dist * 4)
            mob.stretch_to_fit_height(new_height)
            mob.rotate(0.05)

        morph_shape.add_updater(update_morph)

        self.play(
            FadeIn(side_window),
            FadeIn(julia_label),
            FadeIn(c_point),
            FadeIn(morph_shape)
        )
        
        # Animate c_point moving on the Mandelbrot map to drive the morphing
        path_points = [
            mandel_set.get_center() + RIGHT * 0.3,
            mandel_set.get_center() + UP * 0.4,
            mandel_set.get_center() + LEFT * 0.6
        ]
        for p in path_points:
            self.play(c_point.animate.move_to(p), run_time=2, rate_func=rate_functions.linear)
        
        morph_shape.remove_updater(update_morph)
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # "It is the ultimate dictionary of complex dynamics."
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        
        # Display title in gold with Indicate effect
        mb_title_visual = Text("The Mandelbrot Set", font_size=24, color="#FFD700")
        # [RESOLVED Issue 32]: Position at B2-B4 to avoid overlap with side_window at A5
        self.place_in_area(mb_title_visual, "B2", "B4", scale_factor=1.0)
        
        self.play(FadeIn(mb_title_visual))
        self.play(Indicate(mb_title_visual, color="#FFD700"))
        self.wait(2.0)
