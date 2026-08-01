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
        # Setup the layout
        self.setup_layout(
            "The Training Loop: Iterative Improvement", 
            [
                "Training follows a repeating cycle of four steps.", 
                "He performs each task to reduce his error.", 
                "With every cycle, the error point moves lower.", 
                "Finally, we reach the bottom of the error valley.", 
                "Optimization is complete and the robot has learned."
            ]
        )
        
        # --- Create Loop Icons ---
        # Fixed scale factor 0.8 per Issue 43
        predict_icon = Text("Robot", font_size=24, color=WHITE)
        self.place_at_grid(predict_icon, 'B2', scale_factor=0.8)
        
        error_icon = Text("Scorecard", font_size=24, color=WHITE)
        self.place_at_grid(error_icon, 'B5', scale_factor=0.8)
        
        slope_icon = Text("Slope", font_size=24, color=WHITE)
        self.place_at_grid(slope_icon, 'E5', scale_factor=0.8)
        
        step_icon = Text("Step", font_size=24, color=WHITE)
        self.place_at_grid(step_icon, 'E2', scale_factor=0.8)

        # Loop arrows
        arrow_top = Arrow(self.grid['B2'], self.grid['B5'], buff=0.5, color=GRAY)
        arrow_right = Arrow(self.grid['B5'], self.grid['E5'], buff=0.5, color=GRAY)
        arrow_bottom = Arrow(self.grid['E5'], self.grid['E2'], buff=0.5, color=GRAY)
        arrow_left = Arrow(self.grid['E2'], self.grid['B2'], buff=0.5, color=GRAY)
        
        loop_elements = VGroup(predict_icon, error_icon, slope_icon, step_icon, arrow_top, arrow_right, arrow_bottom, arrow_left)

        # --- Create Side-Graph ---
        axes = Axes(
            x_range=[0, 4, 1], 
            y_range=[0, 4, 1], 
            x_length=3.5, 
            y_length=2.5,
            axis_config={"include_tip": False, "font_size": 14}
        )
        def cost_func(x):
            return (x-2)**2 + 0.5
            
        curve = axes.plot(cost_func, x_range=[0, 4], color=BLUE_B)
        dot = Dot(color=RED)
        current_x = 0.4
        dot.move_to(axes.c2p(current_x, cost_func(current_x)))
        
        graph_group = VGroup(axes, curve, dot)
        # Placement per Issue 42
        self.place_in_area(graph_group, 'C2', 'D5', scale_factor=0.9)

        # --- Optimization Complete text ---
        complete_text = Text("Optimization Complete", color="#00FF00", font_size=32)
        complete_text.set_opacity(0)
        # Placement per Issue 44
        self.place_in_area(complete_text, 'F2', 'F5', scale_factor=1.0)

        # Pre-defined x values for steps
        x_steps = [0.4, 1.1, 1.6, 1.85, 2.0]

        def run_cycle_anim(idx, speed=1.0):
            # robot -> scorecard -> slope -> step
            self.play(Indicate(predict_icon, color="#FFFF00"), run_time=0.4/speed)
            self.play(Indicate(error_icon, color="#FFFF00"), run_time=0.4/speed)
            self.play(Indicate(slope_icon, color="#FFFF00"), run_time=0.4/speed)
            target_x = x_steps[idx]
            self.play(
                Indicate(step_icon, color="#FFFF00"),
                dot.animate.move_to(axes.c2p(target_x, cost_func(target_x))),
                run_time=0.6/speed
            )

        # === Animation for Lecture Line 1 ===
        # Color match: Yellow icons
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(FadeIn(loop_elements), FadeIn(graph_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color match: Yellow icons
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        run_cycle_anim(1, speed=1.0)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Color match: Cyan/Blue (Side-graph)
        self.play(self.lecture[2].animate.set_color("#58C4DD"))
        # Run iterations 2 and 3 quickly
        run_cycle_anim(2, speed=2.5)
        run_cycle_anim(3, speed=2.5)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Color match: White/Blue (Bottom of valley)
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))
        run_cycle_anim(4, speed=1.5) # Final step to x=2.0
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Color match: Green text
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        # Flash the text and dot
        self.play(complete_text.animate.set_opacity(1), Flash(dot, color="#00FF00"), run_time=1)
        for _ in range(2):
            self.play(complete_text.animate.set_opacity(0.3), run_time=0.3)
            self.play(complete_text.animate.set_opacity(1), run_time=0.3)
        
        self.wait(2)
