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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Prerequisite Knowledge: The Three Roles", 
            [
                "Every exponential story has three essential parts.",
                "The Base, the Exponent, and the Result.",
                "Imagine a magic slime doubling over time."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Display a 'Growth Machine' icon (#A9A9A9) with three empty slots.
        self.lecture[0].set_color("#A9A9A9")
        
        # Machine body covering the grid area
        machine_body = RoundedRectangle(width=5.5, height=4.8, color="#A9A9A9", fill_opacity=0.05)
        self.place_in_area(machine_body, "A1", "F6")
        
        # Define slots at revised positions (Issues 40 and 41)
        base_slot = Square(side_length=0.9, color="#A9A9A9")
        self.place_at_grid(base_slot, "C2")
        
        exp_slot = Square(side_length=0.9, color="#A9A9A9")
        self.place_at_grid(exp_slot, "C3") # Shifted from C4 to C3 (Issue 41)
        
        res_slot = Square(side_length=0.9, color="#A9A9A9")
        self.place_at_grid(res_slot, "C5") # Shifted from C6 to C5 (Issue 40)
        
        machine_group = VGroup(machine_body, base_slot, exp_slot, res_slot)
        self.play(Create(machine_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fill the first slot with 'Base' slime asset (#FFD700) and show labels.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700")
        
        # Asset integration (Issue 23)
        slime_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slime.svg")
        slime_asset.set_color("#FFD700")
        self.place_at_grid(slime_asset, "C2", scale_factor=0.6)
        
        base_label = Text("Base", font_size=20, color="#FFD700")
        self.place_at_grid(base_label, "B2")
        
        base_multiplier = Text("x2", font_size=24, color="#FFD700")
        self.place_at_grid(base_multiplier, "D2")
        
        # Exponent Label (Issue 41)
        exp_label = Text("Exponent", font_size=20, color="#1E90FF")
        self.place_at_grid(exp_label, "B3", scale_factor=0.8)
        
        # Result Label (Issue 40)
        res_label = Text("Result", font_size=20, color="#ADFF2F")
        self.place_at_grid(res_label, "B5", scale_factor=0.8)
        
        self.play(
            FadeIn(slime_asset),
            Write(base_multiplier),
            Write(base_label),
            Write(exp_label),
            Write(res_label),
            base_slot.animate.set_color("#FFD700")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a clock icon (#1E90FF) ticking 3 times to represent 'Exponent = 3'.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#1E90FF")
        
        # Clock at D3 (Issue 41)
        clock_circle = Circle(radius=0.3, color="#1E90FF")
        self.place_at_grid(clock_circle, "D3")
        
        center_pt = self.grid["D3"]
        clock_hand = Line(center_pt, center_pt + UP * 0.25, color="#1E90FF")
        
        # Use ValueTracker for rotation (Optimization Constraint 10)
        rotation_tracker = ValueTracker(0)
        def update_hand(m):
            angle = -rotation_tracker.get_value() * TAU
            # Sin/Cos order corrected for clockwise rotation starting at UP
            new_end = center_pt + np.array([np.sin(-angle), np.cos(-angle), 0]) * 0.25
            m.set_points_as_corners([center_pt, new_end])
            
        clock_hand.add_updater(update_hand)
        
        exp_val = Text("3", font_size=24, color="#1E90FF")
        self.place_at_grid(exp_val, "C3")
        
        res_val = Text("8", font_size=24, color="#ADFF2F")
        self.place_at_grid(res_val, "C5")
        
        self.play(
            Create(clock_circle), 
            Create(clock_hand),
            exp_slot.animate.set_color("#1E90FF")
        )
        
        # Clock ticks 3 times
        self.play(rotation_tracker.animate.set_value(3), run_time=2, rate_func=linear)
        
        self.play(
            FadeOut(clock_circle, clock_hand),
            Write(exp_val),
            Write(res_val),
            res_slot.animate.set_color("#ADFF2F")
        )
        self.wait(2)
