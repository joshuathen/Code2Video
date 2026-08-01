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
        # Initialization of the teaching scene layout
        title_text = "The Principle of Least Time: Fermat’s Logic"
        lines = [
            'A lifeguard must reach a swimmer as fast as possible.',
            'Multiple paths exist, but only one is the fastest.',
            'The optimal path involves more running and less swimming.',
            'This path minimizes total travel time to the swimmer.',
            'Light follows Fermat’s Principle, taking the path of least time.'
        ]
        self.setup_layout(title_text, lines)

        # Define Colors
        sand_color = "#F4A460"
        water_color = "#1E90FF"
        optimal_color = "#00FF00"
        path_color = "#FFFFFF"

        # Background regions: Top for Sand, Bottom for Water
        sand_rect = Rectangle(width=6.0, height=3.0, fill_color=sand_color, fill_opacity=0.3, stroke_width=0)
        self.place_in_area(sand_rect, "A1", "C6")
        
        water_rect = Rectangle(width=6.0, height=3.0, fill_color=water_color, fill_opacity=0.3, stroke_width=0)
        self.place_in_area(water_rect, "D1", "F6")

        # Medium labels
        sand_lbl = Text("Sand (Fast)", font_size=20, color=sand_color)
        self.place_in_area(sand_lbl, "B5", "B6", scale_factor=0.8)
        
        water_lbl = Text("Water (Slow)", font_size=20, color=water_color)
        self.place_in_area(water_lbl, "E5", "E6", scale_factor=0.8)

        # Key Points: Lifeguard A and Swimmer B
        pos_a = self.grid["A2"]
        pos_b = self.grid["F5"]
        
        dot_a = Dot(pos_a, color=WHITE)
        lbl_a = Text("A", font_size=22).next_to(dot_a, LEFT, buff=0.2)
        dot_b = Dot(pos_b, color=WHITE)
        lbl_b = Text("B", font_size=22).next_to(dot_b, RIGHT, buff=0.2)
        
        # Intersection Points on the interface (between C and D rows)
        y_int = -0.3
        p1 = np.array([3.0, y_int, 0]) # Straight line path intersection
        p2 = np.array([4.0, y_int, 0]) # Optimal path (refracted logic)
        p3 = np.array([5.2, y_int, 0]) # Exaggerated wide path

        # Path display objects
        path1 = VGroup(Line(pos_a, p1), Line(p1, pos_b)).set_stroke(color=path_color, opacity=0.3)
        path2 = VGroup(Line(pos_a, p2), Line(p2, pos_b)).set_stroke(color=path_color, opacity=0.3)
        path3 = VGroup(Line(pos_a, p3), Line(p3, pos_b)).set_stroke(color=path_color, opacity=0.3)

        # Labels
        least_time_lbl = Text("Principle of Least Time", font_size=20, color=WHITE)
        self.place_in_area(least_time_lbl, "F1", "F4", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(sand_rect), FadeIn(water_rect), FadeIn(sand_lbl), FadeIn(water_lbl))
        self.play(FadeIn(dot_a), FadeIn(lbl_a), FadeIn(dot_b), FadeIn(lbl_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        self.play(Create(path1), Create(path2), Create(path3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        self.play(path2.animate.set_stroke(color=optimal_color, opacity=1.0))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        
        # Dots representing movement along candidate paths
        l1 = Dot(pos_a, color=WHITE, radius=0.08)
        l2 = Dot(pos_a, color=optimal_color, radius=0.1)
        l3 = Dot(pos_a, color=WHITE, radius=0.08)

        # Race simulation: Path 2 (Green) is optimized to finish first
        self.play(
            AnimationGroup(
                Succession(l1.animate(run_time=1.46, rate_func=linear).move_to(p1), 
                           l1.animate(run_time=2.91, rate_func=linear).move_to(pos_b)),
                Succession(l2.animate(run_time=1.77, rate_func=linear).move_to(p2), 
                           l2.animate(run_time=2.55, rate_func=linear).move_to(pos_b)),
                Succession(l3.animate(run_time=2.23, rate_func=linear).move_to(p3), 
                           l3.animate(run_time=2.60, rate_func=linear).move_to(pos_b)),
                lag_ratio=0
            )
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        self.play(Write(least_time_lbl))
        self.play(
            path1.animate.set_stroke(opacity=0.05),
            path3.animate.set_stroke(opacity=0.05),
            l1.animate.set_opacity(0),
            l3.animate.set_opacity(0)
        )
        self.wait(3)
