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
            'Chaotic individual dots represent random events in life.',
            'They move together through the funnel of sampling.',
            'They emerge to form a glowing, stable bell curve.',
            'This process transforms total chaos into perfect order.',
            'The Central Limit Theorem governs our complex world.'
        ]
        self.setup_layout("Summary & Key Takeaway", lecture_lines)

        # Common variables
        mu = (self.grid["A1"][0] + self.grid["A6"][0]) / 2
        sigma = 0.7
        y_baseline = self.grid["B1"][1] - 0.2
        num_dots = 100

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Chaotic swarm at the bottom (E1-F6)
        dots = VGroup(*[Dot(radius=0.05, color="#A9A9A9") for _ in range(num_dots)])
        
        for dot in dots:
            dot.move_to([
                np.random.uniform(self.grid["E1"][0]-0.4, self.grid["F6"][0]+0.4),
                np.random.uniform(self.grid["F6"][1]-0.4, self.grid["E1"][1]+0.4),
                0
            ])
            
        def jitter_dots(mob, dt):
            for d in mob:
                d.shift(np.random.uniform(-0.05, 0.05, 3))

        dots.add_updater(jitter_dots)
        self.play(FadeIn(dots))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Funnel Asset [Issue 38]
        funnel = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/funnel.svg", color=BLUE_B)
        self.place_in_area(funnel, "C3", "D4", scale_factor=1.2)
        
        # Label [Issue 57]
        chaos_order_label = Text("Chaos to Order", font_size=20, color=WHITE)
        self.place_in_area(chaos_order_label, "A5", "B6", scale_factor=0.7)
        
        self.play(FadeIn(funnel), Write(chaos_order_label))
        
        # Dots move towards funnel neck
        neck_center = (self.grid["C3"] + self.grid["C4"]) / 2
        dots.remove_updater(jitter_dots)
        
        self.play(
            dots.animate.move_to(neck_center).scale(0.5),
            run_time=2,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Bell Curve formation
        bell_curve = FunctionGraph(
            lambda x: 1.8 * np.exp(-((x - mu)**2) / (2 * (sigma**2))) + y_baseline,
            x_range=[self.grid["A1"][0], self.grid["A6"][0]],
            color=WHITE
        )
        bell_curve_glow = bell_curve.copy().set_stroke(width=10, opacity=0.3, color=WHITE)

        # Target positions for dots along the curve
        move_animations = []
        for dot in dots:
            x_target = np.random.normal(mu, sigma)
            x_target = np.clip(x_target, self.grid["A1"][0], self.grid["A6"][0])
            y_target = 1.8 * np.exp(-((x_target - mu)**2) / (2 * (sigma**2))) + y_baseline
            move_animations.append(dot.animate.move_to([x_target, y_target + np.random.uniform(-0.1, 0.1), 0]).set_color(WHITE).scale(2.0))

        self.play(
            AnimationGroup(*move_animations, lag_ratio=0.01),
            Create(bell_curve),
            FadeIn(bell_curve_glow),
            funnel.animate.set_opacity(0.3),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Highlight stability
        self.play(
            bell_curve_glow.animate.set_stroke(width=20, opacity=0.5),
            dots.animate.set_color(YELLOW),
            run_time=1.5
        )
        self.play(dots.animate.set_color(WHITE), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Pulsing and Glow fill
        glow_overlay = Rectangle(
            width=6, height=6, 
            fill_color=WHITE, fill_opacity=0.1, 
            stroke_width=0
        )
        self.place_in_area(glow_overlay, "A1", "F6")
        
        self.play(
            bell_curve.animate.scale(1.1),
            bell_curve_glow.animate.scale(1.1).set_stroke(opacity=0.4),
            FadeIn(glow_overlay),
            rate_func=there_and_back,
            run_time=4
        )
        self.wait(3)
