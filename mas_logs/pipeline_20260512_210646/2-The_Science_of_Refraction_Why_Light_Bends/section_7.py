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

class Section7Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Lenses use refraction to focus light into sharp images.",
            "Total internal reflection gives diamonds their brilliant sparkle.",
            "Understanding refraction helps the fox finally catch the fish."
        ]
        self.setup_layout("Summary and Synthesis", lecture_lines)

        # Define Colors
        COLOR_LENS = "#E0FFFF"
        COLOR_DIAMOND = "#B9F2FF"
        COLOR_FOX = "#D2691E"
        COLOR_WATER = "#1E90FF"
        COLOR_RAY = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Color matching animation
        self.play(self.lecture[0].animate.set_color(COLOR_LENS))
        
        # Create Convex Lens using Intersection
        c1 = Circle(radius=1.5).shift(LEFT * 1.1)
        c2 = Circle(radius=1.5).shift(RIGHT * 1.1)
        lens = Intersection(c1, c2, color=COLOR_LENS, fill_opacity=0.3)
        # Resolved Issue 51/60: Reposition lens to 'B1'-'D3' (scale 0.5)
        self.place_in_area(lens, "B1", "D3", scale_factor=0.5)
        
        # Parallel incoming rays
        rays_in = VGroup()
        for i in range(-2, 3):
            start_pt = lens.get_left() + LEFT * 1.0 + UP * (i * 0.2)
            end_pt = lens.get_left() + UP * (i * 0.2)
            rays_in.add(Line(start_pt, end_pt, color=COLOR_RAY, stroke_width=2))
            
        # Refracted rays focusing to a single point (focal point)
        focal_pt_pos = lens.get_right() + RIGHT * 1.2
        rays_out = VGroup()
        for i in range(-2, 3):
            start_pt = lens.get_left() + UP * (i * 0.2)
            rays_out.add(Line(start_pt, focal_pt_pos, color=COLOR_RAY, stroke_width=2))

        self.play(FadeIn(lens), Create(rays_in))
        self.play(Create(rays_out))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_DIAMOND)
        )
        
        # Diamond shape
        diamond_pts = [
            [-0.6, 0.4, 0], [0.6, 0.4, 0], [1, 0, 0], [0, -1.2, 0], [-1, 0, 0]
        ]
        diamond = Polygon(*diamond_pts, color=COLOR_DIAMOND, fill_opacity=0.4)
        # Resolved Issue 51/60: Reposition diamond to 'B4'-'D6' (scale 0.5)
        self.place_in_area(diamond, "B4", "D6", scale_factor=0.5)
        
        # Internal light ray path
        p_entry = diamond.get_top() + LEFT * 0.1
        p_ref1 = diamond.get_bottom() + UP * 0.15 + RIGHT * 0.15
        p_ref2 = diamond.get_top() + RIGHT * 0.15
        p_exit = p_ref2 + UP * 0.25 + RIGHT * 0.15
        
        ray_entry = Line(p_entry + UP * 0.25, p_entry, color=COLOR_RAY)
        ray_internal = VMobject().set_points_as_corners([p_entry, p_ref1, p_ref2]).set_color(COLOR_RAY)
        ray_exit = Line(p_ref2, p_exit, color=COLOR_RAY)
        
        # Sparkle effects
        sparkle1 = Star(n=5, outer_radius=0.15, fill_opacity=1, color=WHITE)
        sparkle2 = Star(n=5, outer_radius=0.1, fill_opacity=1, color=WHITE)
        # Resolved Issue 52/60: Adjust sparkle positions to 'A4' and 'A6' (scale 0.7)
        self.place_at_grid(sparkle1, "A4", scale_factor=0.7)
        self.place_at_grid(sparkle2, "A6", scale_factor=0.7)
        
        self.play(
            FadeOut(lens, rays_in, rays_out),
            FadeIn(diamond)
        )
        self.play(Create(ray_entry), run_time=0.5)
        self.play(Create(ray_internal), run_time=1.0)
        self.play(Create(ray_exit), FadeIn(sparkle1, sparkle2), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_FOX)
        )
        
        # Environment
        water_surface = Line(self.grid["C1"], self.grid["C6"], color=COLOR_WATER)
        water_body = Rectangle(width=5.5, height=3.0, fill_color=COLOR_WATER, fill_opacity=0.2, stroke_width=0)
        self.place_in_area(water_body, "D1", "F6", scale_factor=1.0)
        
        # Fox Asset integration
        fox = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/fox.svg")
        fox.set_color(COLOR_FOX)
        # Resolved Issue 53/60: Move fox to 'D2' (scale 0.7)
        self.place_at_grid(fox, "D2", scale_factor=0.7)
        
        # Fish Asset integration
        fish_real = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/fish.svg")
        fish_real.set_color(ORANGE)
        self.place_at_grid(fish_real, "E4", scale_factor=0.4)
        
        fish_apparent = fish_real.copy().set_opacity(0.3)
        self.place_at_grid(fish_apparent, "D4", scale_factor=0.4)
        
        # Aiming arrow (aiming lower than apparent)
        aim_arrow = Arrow(fox.get_center(), fish_real.get_center(), color=COLOR_FOX, buff=0.1)
        
        self.play(
            FadeOut(diamond, ray_entry, ray_internal, ray_exit, sparkle1, sparkle2),
            Create(water_surface), FadeIn(water_body),
            FadeIn(fox), FadeIn(fish_real)
        )
        self.play(FadeIn(fish_apparent))
        self.play(GrowArrow(aim_arrow))
        
        # Fox successfully grabs the fish
        self.play(
            fox.animate.move_to(self.grid["E4"]),
            FadeOut(aim_arrow),
            run_time=1.5
        )
        self.play(
            FadeOut(fish_real),
            fox.animate.set_color(YELLOW).scale(1.1)
        )
        
        self.wait(2)
