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
        # Data from Storyboard
        title_text = "The Power of Abstraction: Why It Matters"
        lecture_lines = [
            "Abstraction lets us solve many problems at once.",
            "One theorem applies to arrows and functions.",
            "Digital signals are processed as abstract vectors.",
            "This 'Master Key' simplifies complex mathematical systems.",
            "We look past form to see underlying structure."
        ]
        
        # Setup
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        PURPLE = "#DDA0DD"
        WHITE_COLOR = "#FFFFFF"
        PALE_YELLOW = "#FFFFE0"
        
        # === Animation for Lecture Line 1 ===
        # Abstraction lets us solve many problems at once.
        self.play(self.lecture[0].animate.set_color(PURPLE))
        
        # Physics icon: Atom-like
        physics_atom = VGroup(
            Ellipse(width=0.6, height=0.15, color=PURPLE).rotate(PI/3),
            Ellipse(width=0.6, height=0.15, color=PURPLE).rotate(-PI/3),
            Dot(radius=0.05, color=PURPLE)
        )
        physics_label = Text("Physics", font_size=16, color=PURPLE)
        physics_icon = VGroup(physics_atom, physics_label).arrange(DOWN, buff=0.1)
        
        # AI icon: Neural net like
        ai_dots = VGroup(
            Dot(color=PURPLE, radius=0.05).shift(UP*0.2 + LEFT*0.1),
            Dot(color=PURPLE, radius=0.05).shift(DOWN*0.2 + LEFT*0.1),
            Dot(color=PURPLE, radius=0.05).shift(RIGHT*0.2)
        )
        ai_lines = VGroup(
            Line(ai_dots[0].get_center(), ai_dots[2].get_center(), stroke_width=1, color=PURPLE),
            Line(ai_dots[1].get_center(), ai_dots[2].get_center(), stroke_width=1, color=PURPLE)
        )
        ai_label = Text("AI", font_size=16, color=PURPLE)
        ai_icon = VGroup(VGroup(ai_dots, ai_lines), ai_label).arrange(DOWN, buff=0.1)
        
        # Signals icon: Sine wave
        signals_wave = VMobject(color=PURPLE)
        signals_wave.set_points_as_corners([np.array([x, 0.15*np.sin(4*PI*x), 0]) for x in np.linspace(-0.3, 0.3, 20)])
        signals_label = Text("Signals", font_size=16, color=PURPLE)
        signals_icon = VGroup(signals_wave, signals_label).arrange(DOWN, buff=0.1)
        
        # Fix for Issue 39: Move to Col 6
        self.place_at_grid(physics_icon, 'A6', scale_factor=0.8)
        self.place_at_grid(ai_icon, 'C6', scale_factor=0.8)
        self.place_at_grid(signals_icon, 'E6', scale_factor=0.8)
        
        self.play(FadeIn(physics_icon), FadeIn(ai_icon), FadeIn(signals_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # One theorem applies to arrows and functions.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE_COLOR)
        )
        
        # Central Vector Space Circle
        vs_circle = Circle(radius=0.7, color=WHITE_COLOR)
        vs_text = Text("Vector Space", font_size=18, color=WHITE_COLOR)
        vector_space = VGroup(vs_circle, vs_text)
        # Fix for Issue 38: Move to C3-D4
        self.place_in_area(vector_space, 'C3', 'D4', scale_factor=0.8) 
        
        # Arrows (avoiding label overlap)
        arrow_p = Arrow(physics_icon.get_left(), vs_circle.get_right(), color=WHITE_COLOR, buff=0.15)
        arrow_a = Arrow(ai_icon.get_left(), vs_circle.get_right(), color=WHITE_COLOR, buff=0.15)
        arrow_s = Arrow(signals_icon.get_left(), vs_circle.get_right(), color=WHITE_COLOR, buff=0.15)
        
        self.play(
            Create(vector_space),
            GrowArrow(arrow_p),
            GrowArrow(arrow_a),
            GrowArrow(arrow_s)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Digital signals are processed as abstract vectors.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(PURPLE)
        )
        
        # Highlight signals icon
        self.play(Indicate(signals_icon, color=PURPLE, scale_factor=1.3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This 'Master Key' simplifies complex mathematical systems.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(PALE_YELLOW)
        )
        
        # Pulsing effect for the central circle using ValueTracker
        pulse_tracker = ValueTracker(1)
        pulse_circle = vs_circle.copy().set_color(PALE_YELLOW)
        
        def update_pulse(m):
            s = pulse_tracker.get_value()
            m.become(vs_circle.copy().set_color(PALE_YELLOW).scale(s))
            m.set_stroke(opacity=1 - (s-1)/0.5)
            
        pulse_circle.add_updater(update_pulse)
        self.add(pulse_circle)
        
        # Animate two pulses
        self.play(pulse_tracker.animate.set_value(1.5), run_time=0.6, rate_func=linear)
        self.play(pulse_tracker.animate.set_value(1), run_time=0.1)
        self.play(pulse_tracker.animate.set_value(1.5), run_time=0.6, rate_func=linear)
        self.play(pulse_tracker.animate.set_value(1), run_time=0.1)
        
        pulse_circle.remove_updater(update_pulse)
        self.remove(pulse_circle)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # We look past form to see underlying structure.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(WHITE_COLOR)
        )
        
        # Emphasize structure: fade outer elements
        self.play(
            VGroup(physics_icon, ai_icon, signals_icon, arrow_p, arrow_a, arrow_s).animate.set_opacity(0.3),
            vs_circle.animate.set_stroke(color=PALE_YELLOW, width=5),
            vs_text.animate.set_color(PALE_YELLOW)
        )
        self.wait(2)
