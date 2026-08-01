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

class Section1Scene(TeachingScene):
    def construct(self):
        # Section Title and Lecture Lines from Shared State
        title_text = "Prerequisite: The Bernoulli Trial"
        lecture_lines = [
            "Imagine a single event with only two possible outcomes.",
            "We call this simple building block a Bernoulli trial.",
            "Success has probability p, while failure has probability q."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors
        FROG_COLOR = "#55FF55"
        FLY_COLOR = "#888888"
        SUCCESS_COLOR = "#55FF55"
        FAILURE_COLOR = "#FF5555"

        # === Animation for Lecture Line 1 ===
        # Imagine a single event with only two possible outcomes.
        self.lecture[0].set_color(YELLOW)
        
        # Load Assets (Issue 22)
        frog = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/frog.svg")
        frog.set_color(FROG_COLOR)
        fly = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fly.svg")
        fly.set_color(FLY_COLOR)

        # Position objects in Col 3 and 5 (L003 buffer)
        self.place_at_grid(frog, "C3", scale_factor=0.6)
        self.place_at_grid(fly, "C5", scale_factor=0.4)
        
        self.play(FadeIn(frog, shift=RIGHT), FadeIn(fly, shift=LEFT))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We call this simple building block a Bernoulli trial.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Label the trial in Row A (Area-Positioning Rule)
        trial_label = Text("Bernoulli Trial", font_size=28, color=BLUE)
        self.place_in_area(trial_label, "A3", "A5", scale_factor=0.8)
        
        self.play(Write(trial_label))
        # Highlight the trial components using Indicate (L004)
        self.play(Indicate(frog, color=YELLOW), Indicate(fly, color=YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Success has probability p, while failure has probability q.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Prepare success/failure labels and formulas
        success_label = Text("Success", font_size=24, color=SUCCESS_COLOR)
        prob_p = MathTex("p", color=SUCCESS_COLOR)
        failure_label = Text("Failure", font_size=24, color=FAILURE_COLOR)
        prob_q = MathTex("q = 1 - p", color=FAILURE_COLOR)
        
        # Position labels using VideoCritic fixes (Issues 25, 26, 27)
        # Position Success labels horizontally relative to Frog (Issue 25)
        self.place_at_grid(success_label, "B3", scale_factor=0.8)
        self.place_at_grid(prob_p, "B2", scale_factor=0.8)
        
        # Position Failure labels near miss path (Issues 26, 27)
        self.place_at_grid(failure_label, "E6", scale_factor=0.8)
        self.place_in_area(prob_q, "E4", "E5", scale_factor=0.8)
        
        frog_home_pos = frog.get_center().copy()
        
        # Part A: Successful Catch
        self.play(
            frog.animate.move_to(fly.get_center()),
            FadeOut(fly, shift=UP),
            run_time=1
        )
        self.play(
            Write(success_label),
            Write(prob_p)
        )
        self.wait(1.5)
        
        # Reset for Failure Demo
        self.play(
            frog.animate.move_to(frog_home_pos),
            FadeIn(fly),
            success_label.animate.set_fill(opacity=0.3),
            prob_p.animate.set_fill(opacity=0.3),
            run_time=0.8
        )
        
        # Part B: Failure Miss
        # Frog jumps but misses target (moves to D5 instead of C5)
        miss_pos = self.grid["D5"]
        self.play(
            frog.animate.move_to(miss_pos),
            run_time=1
        )
        self.play(
            Write(failure_label),
            Write(prob_q)
        )
        self.wait(2)
