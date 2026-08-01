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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Nature often seems chaotic and completely random.',
            'But look closer at these ten thousand squirrels.',
            'Their individual weights form a messy, jagged mountain.',
            'Yet, group averages reveal a mysterious, hidden order.',
            "Why does this perfect 'Bell Curve' always appear?"
        ]
        self.setup_layout("The Mystery of Predictability", lecture_lines)
        
        # Colors from description
        COLOR_CHAOS = WHITE
        COLOR_HIGHLIGHT_DOT = "#FF8A65"
        COLOR_JAGGED = "#FF5252"
        COLOR_BELL = "#66BB6A"

        # Helper for jagged distribution
        def get_jagged_weight():
            choice = np.random.rand()
            if choice < 0.3: return np.random.normal(2, 0.4)
            if choice < 0.6: return np.random.normal(5, 0.3)
            return np.random.normal(8, 0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_CHAOS)
        
        # Display 800 small dots jittering in a central box (performance compromise for 1000)
        chaos_dots = VGroup(*[
            Dot(radius=0.015, color=COLOR_CHAOS, fill_opacity=0.7)
            for _ in range(800)
        ])
        for dot in chaos_dots:
            dot.move_to([
                np.random.uniform(1.0, 5.5),
                np.random.uniform(-1.5, 1.5),
                0
            ])
            
        def jitter(mob, dt):
            mob.shift(np.random.uniform(-0.03, 0.03, 3))
        
        for dot in chaos_dots:
            dot.add_updater(jitter)
            
        self.add(chaos_dots)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT_DOT)
        
        # Highlight dots in #FF8A65
        highlight_indices = np.random.choice(len(chaos_dots), 50, replace=False)
        self.play(*[
            chaos_dots[idx].animate.set_color(COLOR_HIGHLIGHT_DOT).scale(2)
            for idx in highlight_indices
        ], run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_JAGGED)
        
        # Stop jitter
        for dot in chaos_dots:
            dot.remove_updater(jitter)
            
        # Top axis - Issue 32 fix: Row B
        top_axis = NumberLine(x_range=[0, 10, 1], length=5, include_tip=True, color=GRAY)
        self.place_in_area(top_axis, 'B1', 'B6')
        
        # Labels - Issue 34 fix: Row A
        tiny_label = Text("Tiny", font_size=18, color=COLOR_JAGGED)
        huge_label = Text("Huge", font_size=18, color=COLOR_JAGGED)
        self.place_at_grid(tiny_label, 'A1', scale_factor=0.8)
        self.place_at_grid(huge_label, 'A6', scale_factor=0.8)
        
        # Label for distribution
        top_label = Text("Individual Weights", font_size=18)
        self.place_at_grid(top_label, 'A3', scale_factor=1.0)
        
        # Move dots to form jagged mountain
        dot_anims = []
        for dot in chaos_dots:
            w = get_jagged_weight()
            w = max(0.5, min(9.5, w))
            h = np.random.uniform(0.1, 0.8)
            target = top_axis.number_to_point(w) + UP * h
            dot_anims.append(dot.animate.move_to(target).set_color(WHITE).scale(0.5))
            
        # Jagged line plot
        points = []
        for x in np.linspace(0.5, 9.5, 25):
            y_val = 0.4 * np.exp(-0.5*((x-2)/0.4)**2) + 0.6 * np.exp(-0.5*((x-5)/0.3)**2) + 0.3 * np.exp(-0.5*((x-8)/0.5)**2)
            points.append(top_axis.number_to_point(x) + UP * (y_val * 1.5 + 0.1))
        
        jagged_plot = VMobject(color=COLOR_JAGGED, stroke_width=3)
        jagged_plot.set_points_as_corners(points)
        
        self.play(
            FadeIn(top_axis), FadeIn(tiny_label), FadeIn(huge_label), FadeIn(top_label),
            *dot_anims,
            run_time=2
        )
        self.play(Create(jagged_plot), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_BELL)
        
        # Bottom axis - Issue 33 fix: Row E
        bottom_axis = NumberLine(x_range=[0, 10, 1], length=5, include_tip=True, color=GRAY)
        self.place_in_area(bottom_axis, 'E1', 'E6')
        
        bottom_label = Text("Sample Means", font_size=18)
        self.place_at_grid(bottom_label, 'D3', scale_factor=1.0)
        
        self.play(FadeIn(bottom_axis), FadeIn(bottom_label))
        
        # Group dots (30 dots merge into one)
        sample_size = 30
        
        # Take 10 samples to show animation clearly
        for _ in range(10):
            indices = np.random.choice(len(chaos_dots), sample_size, replace=False)
            subset = VGroup(*[chaos_dots[i] for i in indices])
            
            # Calculate mean
            weights = [top_axis.point_to_number(d.get_center()) for d in subset]
            m_val = np.mean(weights)
            
            target_dot = Dot(radius=0.06, color=COLOR_BELL)
            target_dot.move_to(bottom_axis.number_to_point(m_val) + UP*0.1)
            
            self.play(
                subset.animate.set_color(COLOR_BELL).scale(1.2),
                run_time=0.2
            )
            self.play(
                ReplacementTransform(subset, target_dot),
                run_time=0.4
            )
            
        # Add many more means to show distribution forming
        more_dots = VGroup()
        for _ in range(150):
            s_weights = [get_jagged_weight() for _ in range(sample_size)]
            mv = np.mean(s_weights)
            mv = max(0.5, min(9.5, mv))
            p = bottom_axis.number_to_point(mv)
            count = sum(1 for d in more_dots if abs(d.get_center()[0] - p[0]) < 0.1)
            p += UP * (0.1 + count * 0.04)
            more_dots.add(Dot(radius=0.04, color=COLOR_BELL).move_to(p))
            
        self.play(FadeIn(more_dots, lag_ratio=0.005), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_BELL)
        
        # Smooth Bell Curve setup
        bell_axes = Axes(x_range=[0, 10, 1], y_range=[0, 1, 0.2], tips=False)
        self.place_in_area(bell_axes, 'E1', 'E6', scale_factor=0.8)
        
        bell_curve = bell_axes.plot(
            lambda x: 0.8 * np.exp(-0.5 * ((x - 5.0) / 0.7)**2),
            color=COLOR_BELL, x_range=[2, 8]
        )
        
        # Morph jagged #FF5252 line to smooth #66BB6A Bell Curve
        # First align the jagged plot to the same area as the bell curve
        self.play(
            jagged_plot.animate.move_to(bell_axes.get_center() + UP*0.5),
            run_time=1
        )
        self.play(ReplacementTransform(jagged_plot, bell_curve), run_time=2)
        self.play(bell_curve.animate.set_stroke(width=6), run_time=1)
        self.wait(3)
