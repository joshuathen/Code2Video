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

class Section4Scene(TeachingScene):
    def construct(self):
        # Data from storyboard and script
        title_text = "The 'Crust' Phenomenon"
        lecture_lines = [
            "- High-dimensional spheres are mostly made of crust.",
            "- Almost all volume lies near the outer surface.",
            "- The interior of the sphere is effectively empty."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        ORANGE_COLOR = "#FFA500"
        
        # Assets
        sphere_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        orange_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg"

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(ORANGE_COLOR))
        
        # Circle with a shaded outer ring (crust)
        circle_outline = Circle(radius=1.5, color=WHITE, stroke_width=2)
        crust_2d = Annulus(inner_radius=1.3, outer_radius=1.5, color=ORANGE_COLOR, fill_opacity=0.6, stroke_width=0)
        
        viz_group = VGroup(crust_2d, circle_outline)
        self.place_in_area(viz_group, 'B2', 'E5', scale_factor=1.0)
        center_point = viz_group.get_center().copy()
        
        self.play(Create(circle_outline), FadeIn(crust_2d))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(ORANGE_COLOR)
        )
        
        # 3D sphere visualization with a shrinking inner core
        sphere_svg = SVGMobject(sphere_asset_path)
        sphere_svg.set_color(WHITE).set_opacity(0.3)
        sphere_svg.scale_to_fit_width(3.0)
        sphere_svg.move_to(center_point)
        
        # shell_bg represents the total sphere volume (mostly orange/crust at high n)
        # core_black represents the interior volume which becomes negligible
        shell_bg = Circle(radius=1.5, color=ORANGE_COLOR, fill_opacity=1, stroke_width=0).move_to(center_point)
        core_black = Circle(radius=1.45, color=BLACK, fill_opacity=1, stroke_width=0).move_to(center_point)
        
        n_tracker = ValueTracker(3)
        
        def core_updater(m):
            n = n_tracker.get_value()
            # Visual heuristic for shrinking interior: R_inner = R * (0.95)^(n/5)
            new_radius = 1.5 * (0.95 ** (n / 5))
            m.set_width(max(0.1, 2 * new_radius))
            m.move_to(center_point)

        core_black.add_updater(core_updater)
        
        # Dimension counter
        n_label = Text("n = ", font_size=24, color=WHITE)
        n_val = Integer(3, font_size=24, color=WHITE).next_to(n_label, RIGHT)
        n_group = VGroup(n_label, n_val)
        
        # ISSUE 37 FIX: Adjusted position to A4 to avoid overlap
        self.place_at_grid(n_group, 'A4', scale_factor=0.8)
        
        n_val.add_updater(lambda m: m.set_value(int(n_tracker.get_value())))

        # Transition to the interactive shell model
        self.play(
            FadeOut(crust_2d),
            FadeIn(shell_bg),
            FadeIn(core_black),
            FadeIn(sphere_svg),
            FadeIn(n_group)
        )
        
        # Animate dimension increase and corresponding core shrinkage
        self.play(n_tracker.animate.set_value(100), run_time=5, rate_func=smooth)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(ORANGE_COLOR)
        )
        
        # Replace shell model with orange icon to represent the 'orange' analogy
        orange_svg = SVGMobject(orange_asset_path)
        orange_svg.set_color(ORANGE_COLOR)
        orange_svg.scale_to_fit_width(3.0)
        orange_svg.move_to(center_point)
        
        # Tiny dot to show the only remaining interior part
        center_dot = Dot(center_point, radius=0.04, color=WHITE)
        label_empty = Text("Mostly empty interior", font_size=20, color=WHITE)
        
        # ISSUE 38 FIX: Adjusted position to F4 to avoid overlap
        self.place_at_grid(label_empty, 'F4', scale_factor=0.8)
        
        arrow = Arrow(start=label_empty.get_top(), end=center_point, buff=0.2, color=WHITE)

        # Final cleanup of updaters to prevent performance issues
        core_black.clear_updaters()
        n_val.clear_updaters()
        
        self.play(
            FadeOut(shell_bg),
            FadeOut(core_black),
            FadeOut(sphere_svg),
            FadeIn(orange_svg),
            FadeIn(center_dot),
            Create(arrow),
            FadeIn(label_empty)
        )
        
        # Accentuate final state
        self.play(n_val.animate.set_color(ORANGE_COLOR).scale(1.2))
        self.wait(2)
