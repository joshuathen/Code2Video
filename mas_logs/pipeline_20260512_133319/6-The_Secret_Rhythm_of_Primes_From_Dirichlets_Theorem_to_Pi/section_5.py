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
        # Initial Setup
        lines = [
            "Dirichlet ensures infinity, Chebyshev shows bias, and Pi emerges.",
            "Primes are not just random; they possess a hidden rhythm.",
            "Number theory unveils the deep structure of our mathematical universe."
        ]
        self.setup_layout("Conclusion: The Hidden Order", lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Dirichlet Icon (Infinity) - Issue 45 fix: scale 0.8 at B2
        dirichlet_icon = Text("∞", color="#FFFF00")
        self.place_at_grid(dirichlet_icon, "B2", scale_factor=0.8)
        
        # Chebyshev / Conveyor Montage - Issue 33, 51 integration
        conveyor_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/conveyor.svg").set_color(TEAL)
        belt_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/belt.svg").set_color(BLUE_C)
        race_line1 = Line(LEFT*0.5, RIGHT*0.5, color=BLUE).shift(UP*0.05)
        race_line2 = Line(LEFT*0.5, RIGHT*0.5, color=RED).shift(DOWN*0.05)
        race_graph = VGroup(race_line1, race_line2)
        chebyshev_montage = VGroup(conveyor_svg, belt_svg, race_graph).arrange(DOWN, buff=0.15)
        self.place_at_grid(chebyshev_montage, "B5", scale_factor=0.6)
        
        # Pi Icon (Circle with Pi) - Issue 46 fix: place at E5
        circle_icon = Circle(radius=0.5, color=WHITE)
        pi_symbol = Text("π", color=WHITE)
        pi_icon = VGroup(circle_icon, pi_symbol)
        self.place_at_grid(pi_icon, "E5", scale_factor=0.8)

        self.play(
            FadeIn(dirichlet_icon),
            FadeIn(chebyshev_montage),
            FadeIn(pi_icon)
        )
        self.wait(1)

        # Merging into center
        merge_point = self.grid["C4"]
        self.play(
            dirichlet_icon.animate.move_to(merge_point).set_opacity(0),
            chebyshev_montage.animate.move_to(merge_point).set_opacity(0),
            pi_icon.animate.move_to(merge_point).set_opacity(0),
            run_time=1.5
        )

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Ulam Spiral (Simplified dot pattern)
        # 4n+1: Blue, 4n+3: Red, Others: Gray
        dot_colors = [
            "#555555", "#555555", "#FF0000", "#555555", "#0000FF", 
            "#555555", "#FF0000", "#555555", "#555555", "#555555", 
            "#FF0000", "#555555", "#0000FF", "#555555", "#FF0000"
        ]
        
        # Coordinates for a tiny spiral
        spiral_coords = [
            (0,0), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), 
            (1,-1), (2,-1), (2,0), (2,1), (2,2), (1,2), (0,2)
        ]
        
        spiral_dots = VGroup()
        for i, (dx, dy) in enumerate(spiral_coords):
            dot = Dot(point=[dx*0.4, dy*0.4, 0], color=dot_colors[i], radius=0.08)
            spiral_dots.add(dot)
            
        self.place_in_area(spiral_dots, "B3", "E5", scale_factor=0.8)
        
        self.play(
            LaggedStart(*[FadeIn(dot) for dot in spiral_dots], lag_ratio=0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Final unified truth
        main_circle = Circle(radius=1.8, color=WHITE).set_stroke(width=2, opacity=0.3)
        self.place_in_area(main_circle, "A2", "F6", scale_factor=1.0)
        
        # Issue 47 fix: Final text at F2-F5 to avoid cluttering with spiral_dots
        final_text = Text("The Secret Rhythm of Primes", color=WHITE, font_size=24)
        self.place_in_area(final_text, "F2", "F5", scale_factor=0.8)

        self.play(
            spiral_dots.animate.set_opacity(0.3),
            FadeIn(main_circle),
            FadeIn(final_text),
            run_time=2
        )
        
        # Shimmer effect (simple scale pulse)
        self.play(
            main_circle.animate.scale(1.05).set_opacity(0.6),
            rate_func=there_and_back,
            run_time=2
        )
        
        self.wait(2)
        self.lecture[2].set_color(WHITE)
        self.wait(1)
