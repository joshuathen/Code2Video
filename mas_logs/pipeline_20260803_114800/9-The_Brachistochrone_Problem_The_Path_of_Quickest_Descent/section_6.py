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
        # Initialize Layout
        self.setup_layout(
            "Real-World Application & Summary",
            [
                "- From physics puzzles to modern engineering and roller coasters.",
                "- The cycloid principles optimize speed and passenger thrill.",
                "- Calculus of variations proves the beauty of the quickest descent."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Visual comparison of the straight line (#FFFFFF) vs. the optimal cycloid descent (#00FFFF).
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        start_pt = self.grid["C1"]
        end_pt = self.grid["F5"]
        
        # Straight path
        straight_path = Line(start_pt, end_pt, color="#FFFFFF")
        
        # Cycloid-like path (Steep start, transitioning to horizontal)
        # Using CubicBezier for performance and predictable shape
        cycloid_path = CubicBezier(
            start_pt, 
            start_pt + 3 * DOWN, 
            end_pt + 1.5 * LEFT, 
            end_pt, 
            color="#00FFFF"
        )
        
        # Racing dots
        ball_straight = Dot(color="#FFFFFF")
        ball_cycloid = Dot(color="#00FFFF")
        
        # Labels
        label_s = Text("Straight Line", font_size=16, color="#FFFFFF")
        label_c = Text("Optimal Cycloid", font_size=16, color="#00FFFF")
        
        # [Fix for Issue 38 & 39]
        self.place_at_grid(label_s, "B5", scale_factor=0.9)
        self.place_at_grid(label_c, "B2", scale_factor=0.9)

        self.play(
            Create(straight_path),
            Create(cycloid_path),
            Write(label_s),
            Write(label_c)
        )
        
        # The race animation
        # The cycloid path dot finishes significantly faster
        self.play(
            MoveAlongPath(ball_cycloid, cycloid_path, rate_func=linear, run_time=1.8),
            MoveAlongPath(ball_straight, straight_path, rate_func=linear, run_time=2.6)
        )
        self.wait(1)

        # Cleanup for next segment
        self.play(
            FadeOut(straight_path),
            FadeOut(ball_straight),
            FadeOut(ball_cycloid),
            FadeOut(label_s),
            FadeOut(label_c),
            FadeOut(cycloid_path)
        )

        # === Animation for Lecture Line 2 ===
        # Overlay a cycloid curve (#FFFF00) onto a skatepark bowl silhouette (#333333).
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        
        # Bowl silhouette using an arc
        bowl_silhouette = Arc(radius=2.0, start_angle=PI, angle=PI, color="#333333", stroke_width=10)
        self.place_in_area(bowl_silhouette, "C1", "F6")
        
        # Matching cycloid curve to overlay
        # Parametric cycloid: x = a(t - sin t), y = a(1 - cos t)
        cycloid_overlay = ParametricFunction(
            lambda t: np.array([1.2 * (t - np.sin(t)), -1.2 * (1 - np.cos(t)), 0]),
            t_range=[0, TAU],
            color="#FFFF00"
        )
        # Shift and scale to align with bowl area
        self.place_in_area(cycloid_overlay, "C1", "F6", scale_factor=0.8)

        self.play(Create(bowl_silhouette))
        self.play(Create(cycloid_overlay))
        self.wait(2)
        
        self.play(
            FadeOut(bowl_silhouette),
            FadeOut(cycloid_overlay)
        )

        # === Animation for Lecture Line 3 ===
        # The text 'The Path of Quickest Descent' fades in (#FFFFFF).
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE)
        )
        
        final_summary = Text("The Path of Quickest Descent", color=WHITE, font_size=32)
        
        # [Fix for Issue 37]
        self.place_in_area(final_summary, "C2", "D5", scale_factor=0.8)
        
        self.play(FadeIn(final_summary))
        self.wait(4)
