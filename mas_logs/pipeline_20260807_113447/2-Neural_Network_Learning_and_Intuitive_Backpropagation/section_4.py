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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Loss Function: Measuring the 'Oops'",
            [
                "Loss measures the distance from the truth.",
                "We visualize error as a hilly landscape.",
                "The lowest point represents zero prediction error."
            ]
        )
        
        # Assets
        LANDSCAPE_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/landscape.svg"
        BALL_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg"

        # === Animation for Lecture Line 1 ===
        # Show a bar graph labeled 'Error Meter' spiking to a high value (#FF4500).
        line1_color = "#FF4500"
        self.play(self.lecture[0].animate.set_color(line1_color))
        
        error_meter_label = Text("Error Meter", font_size=24, color=WHITE)
        # Resolved Issue 31: Use area positioning for multi-word label
        self.place_in_area(error_meter_label, "B1", "B3", scale_factor=0.8)
        
        bar_bg = Rectangle(height=2.5, width=0.8, stroke_color=WHITE, stroke_width=2, fill_opacity=0.1)
        self.place_in_area(bar_bg, "C2", "E2")
        
        bar_height_tracker = ValueTracker(0.1)
        bar = Rectangle(
            height=0.1, 
            width=0.7, 
            fill_color=line1_color, 
            fill_opacity=0.8, 
            stroke_width=0
        )
        # Position bar at the bottom of the background
        bar.move_to(bar_bg.get_bottom(), aligned_edge=DOWN)
        
        def update_bar(m):
            new_height = bar_height_tracker.get_value()
            m.stretch_to_fit_height(max(0.01, new_height))
            m.move_to(bar_bg.get_bottom(), aligned_edge=DOWN)
            
        bar.add_updater(update_bar)
        
        self.play(Create(bar_bg), Write(error_meter_label))
        self.add(bar)
        self.play(bar_height_tracker.animate.set_value(2.3), run_time=2, rate_func=slow_into)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visualize a 2D curvy line (the landscape) [Asset: ...] with a ball [Asset: ...] sitting on a peak.
        line2_color = "#87CEEB"
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(line2_color)
        )
        
        # Resolved Issue 24: Integrated landscape and ball assets
        landscape = SVGMobject(LANDSCAPE_PATH, color=line2_color)
        self.place_in_area(landscape, "C4", "E6", scale_factor=1.5)
        
        ball = SVGMobject(BALL_PATH, color=WHITE)
        # Place ball at a 'peak' (C4)
        self.place_at_grid(ball, "C4", scale_factor=0.4)
        
        self.play(DrawBorderThenFill(landscape))
        self.play(FadeIn(ball))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the valley of the landscape in green (#00FF00) as the 'Goal'.
        line3_color = "#00FF00"
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(line3_color)
        )
        
        # Valley position (E6 corresponds to the right edge/bottom of landscape area)
        valley_pos = self.grid["E6"]
        
        goal_highlight = Circle(radius=0.3, color=line3_color, stroke_width=4)
        goal_highlight.move_to(valley_pos)
        
        goal_label = Text("Goal", font_size=24, color=line3_color)
        # Resolved Issue 32: Positioned goal label at F6 for better alignment
        self.place_at_grid(goal_label, "F6", scale_factor=0.8)
        
        # Animate ball moving to the valley
        self.play(ball.animate.move_to(valley_pos), run_time=2, rate_func=smooth)
        self.play(Create(goal_highlight), Write(goal_label))
        self.play(Indicate(goal_highlight, color=line3_color))
        self.wait(2)
