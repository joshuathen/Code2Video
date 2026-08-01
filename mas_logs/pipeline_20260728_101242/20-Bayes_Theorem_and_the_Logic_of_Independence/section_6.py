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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize layout
        title_text = "Summary: Thinking Like a Bayesian"
        lecture_lines = [
            "Independence ignores evidence; dependence requires updating.",
            "Bayes' theorem provides the mathematical engine for updates.",
            "Use data to move from uncertainty to truth."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define colors for lecture steps
        color_step1 = BLUE_C
        color_step2 = GREEN_C
        color_step3 = YELLOW_C

        # === Animation for Lecture Line 1 ===
        # Fix for Issue 44: Shift equation elements to A1-A5 to avoid edge cut-off
        self.play(self.lecture[0].animate.set_color(color_step1))
        
        initial_belief = Text("Initial Belief", font_size=20, color=color_step1)
        plus_sign = Text("+", font_size=30, color=WHITE)
        new_data = Text("New Data", font_size=20, color=color_step1)
        flow_arrow = Arrow(LEFT, RIGHT, color=WHITE, buff=0.1)
        refined_truth = Text("Refined Truth", font_size=20, color=color_step1)
        
        # Positioning using grid (A1-A5)
        self.place_at_grid(initial_belief, "A1", scale_factor=0.8)
        self.place_at_grid(plus_sign, "A2", scale_factor=1.0)
        self.place_at_grid(new_data, "A3", scale_factor=0.8)
        self.place_at_grid(flow_arrow, "A4", scale_factor=0.7)
        self.place_at_grid(refined_truth, "A5", scale_factor=0.8)
        
        flow_diagram = VGroup(initial_belief, plus_sign, new_data, flow_arrow, refined_truth)
        self.play(FadeIn(flow_diagram))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_step2)
        )
        
        # Integration of Assets (Issue 29)
        # Gauge Visual
        gauge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gauge.svg")
        self.place_in_area(gauge, "D2", "F5", scale_factor=1.2)
        gauge_center = gauge.get_center() + DOWN * 0.5 # Adjustment for gauge pivot point
        
        # Labels for the gauge
        label_uncertain = Text("Uncertain", font_size=16, color=RED)
        label_certain = Text("Certainty", font_size=16, color=GREEN)
        self.place_at_grid(label_uncertain, "F1", scale_factor=0.8)
        self.place_at_grid(label_certain, "F6", scale_factor=0.8)
        
        # Needle Asset
        needle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/needle.svg")
        needle.set_color(color_step2)
        # Assume needle pivot is at the center of the SVG and it points 'up'
        # We need to shift the pivot to the bottom
        needle.move_to(gauge_center, aligned_edge=DOWN)
        
        # ValueTracker for the needle angle (PI is left, 0 is right)
        angle_tracker = ValueTracker(PI)
        self.last_angle = PI

        def needle_updater(mob):
            new_angle = angle_tracker.get_value()
            # Rotate relative to previous state to avoid cumulative error
            mob.rotate(new_angle - self.last_angle, about_point=gauge_center)
            self.last_angle = new_angle
            
        needle.add_updater(needle_updater)
        
        self.play(FadeIn(gauge), FadeIn(label_uncertain), FadeIn(label_certain), FadeIn(needle))
        
        # Clues that cause updates (Issue 45: Using areas)
        clue_1 = Text("Clue: Fingerprints Found", font_size=18, color=color_step2)
        self.place_in_area(clue_1, "C1", "C3", scale_factor=0.8)
        
        self.play(FadeIn(clue_1))
        self.play(angle_tracker.animate.set_value(PI * 0.7), run_time=1)
        self.wait(0.5)

        clue_2 = Text("Clue: Alibi Disproven", font_size=18, color=color_step2)
        self.place_in_area(clue_2, "C4", "C6", scale_factor=0.8)
        
        self.play(FadeIn(clue_2))
        self.play(angle_tracker.animate.set_value(PI * 0.3), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_step3)
        )
        
        # Final update to Certainty
        self.play(angle_tracker.animate.set_value(0), run_time=1.5)
        
        # Final Message (Issue 46: Expanding area)
        final_message = Text("Think Bayesian", font_size=40, color="#FFFF00")
        self.place_in_area(final_message, "B1", "B6", scale_factor=1.0)
        
        self.play(Write(final_message))
        self.play(final_message.animate.scale(1.2), rate_func=there_and_back)
        self.wait(2)
