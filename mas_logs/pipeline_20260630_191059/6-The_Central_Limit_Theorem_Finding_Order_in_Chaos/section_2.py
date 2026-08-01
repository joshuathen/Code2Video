from manim import *
import numpy as np
import random

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
        # Data from storyboard
        title = "The Action: The Sampling Process"
        lines = [
            "Sam catches a random sample of thirty squirrels.",
            "He calculates the average weight for this group.",
            "This average is recorded as one sample mean.",
            "Sam repeats this process many times over.",
            "He creates a collection of hundreds of means."
        ]
        
        self.setup_layout(title, lines)

        # Colors
        COLOR_POP = "#666666" # Gray population
        COLOR_SAMPLE = "#00FF00" # Green sample
        COLOR_MEAN = "#FFFF00" # Yellow sticky notes
        COLOR_BOX = "#3399FF" # Blue calculator

        # === Animation for Lecture Line 1 ===
        # Create population circles
        population = VGroup(*[Circle(radius=0.1, color=COLOR_POP, fill_opacity=0.5) for _ in range(60)])
        for i, circle in enumerate(population):
            row_idx = i // 20
            row = ["A", "B", "C"][row_idx]
            col_idx = (i % 20) // 4
            col = str(col_idx + 1)
            # Add some jitter
            jitter = np.array([random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), 0])
            circle.move_to(self.grid[f"{row}{col}"] + jitter)
        
        self.play(FadeIn(population), self.lecture[0].animate.set_color(COLOR_SAMPLE))
        
        # Select 30 random squirrels
        sample_indices = random.sample(range(len(population)), 30)
        sample_circles = VGroup(*[population[i] for i in sample_indices])
        
        # FIX: Move sample_label to A1 (Issue 21)
        sample_label = Text("Sample (n=30)", font_size=18, color=COLOR_SAMPLE)
        self.place_at_grid(sample_label, 'A1', scale_factor=0.8)
        
        self.play(
            *[obj.animate.set_color(COLOR_SAMPLE).set_fill(COLOR_SAMPLE, 0.8) for obj in sample_circles],
            FadeIn(sample_label),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # FIX: Use SVG for calculator (Issue 17) and move to D3 (Issue 23)
        calculator_svg = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/calculator.svg", color=COLOR_BOX)
        self.place_at_grid(calculator_svg, 'D3', scale_factor=0.9)
        
        calc_label = Text("Calculate Mean", font_size=16, color=WHITE).next_to(calculator_svg, DOWN, buff=0.1)
        calculator = VGroup(calculator_svg, calc_label)

        self.play(FadeIn(calculator), self.lecture[1].animate.set_color(COLOR_BOX))
        
        # Move sample circles to calculator
        self.play(
            sample_circles.animate.scale(0.2).move_to(calculator_svg.get_center()),
            FadeOut(sample_label),
            run_time=1.5
        )
        self.play(FadeOut(sample_circles))

        # === Animation for Lecture Line 3 ===
        # Output Sample Mean
        mean_val_text = Text("x̄ = 1.2", font_size=24, color=COLOR_MEAN)
        self.place_at_grid(mean_val_text, "D5")
        
        self.play(Write(mean_val_text), self.lecture[2].animate.set_color(COLOR_MEAN))
        
        # Transform to sticky note
        # FIX: Move sticky_note to F2 (Issue 22)
        sticky_note = Square(side_length=0.3, fill_color=COLOR_MEAN, fill_opacity=1.0, stroke_width=1)
        self.place_at_grid(sticky_note, 'F2', scale_factor=0.6)
        
        self.play(
            Transform(mean_val_text, sticky_note),
            run_time=1
        )
        self.remove(mean_val_text)
        self.add(sticky_note)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_MEAN))
        
        # Repopulate sample circles for repeat visual
        for obj in population:
            obj.set_color(COLOR_POP).set_fill(COLOR_POP, 0.5)
        
        sticky_notes = VGroup(sticky_note)
        
        # Fast repetition loop
        # Start repeat positioning from F3 onwards to avoid overlapping the initial note at F2
        for i in range(1, 15):
            # Flash population
            temp_indices = random.sample(range(len(population)), 30)
            flash_circles = VGroup(*[population[idx] for idx in temp_indices])
            
            # Simple fast movement
            new_note = Square(side_length=0.3, fill_color=COLOR_MEAN, fill_opacity=1.0, stroke_width=1).scale(0.6)
            
            # Repetition positioning: start at F3, go to F6, then E2-E6
            pos_idx = i + 2 # offset by 2 because F2 is the first one
            col_num = (pos_idx % 6) + 1
            row_char = "F" if pos_idx < 6 else "E"
            
            # Ensure col_num is at least 1 and within bounds
            target_grid = f"{row_char}{col_num}"
            new_note.move_to(self.grid[target_grid])
            
            self.play(
                Flash(calculator_svg, color=COLOR_MEAN, flash_radius=0.5, run_time=0.2),
                FadeIn(new_note, shift=UP * 0.2, run_time=0.2),
                flash_circles.animate(run_time=0.1).set_color(COLOR_SAMPLE),
            )
            self.play(flash_circles.animate(run_time=0.1).set_color(COLOR_POP))
            sticky_notes.add(new_note)

        # === Animation for Lecture Line 5 ===
        self.play(
            FadeOut(population),
            FadeOut(calculator),
            self.lecture[4].animate.set_color(COLOR_MEAN)
        )
        
        # Final stack of many notes
        bins = [1, 3, 6, 10, 8, 5, 4, 2, 1] 
        final_notes = VGroup()
        
        # Create a floor line - shifting slightly to the right to align with F2-F6 area
        floor = Line(self.grid["F2"] + LEFT*0.5, self.grid["F6"] + RIGHT*1.0, color=WHITE)
        self.play(Create(floor))

        all_animations = []
        note_idx = 0
        for col_idx, count in enumerate(bins):
            for row_idx in range(count):
                if note_idx < len(sticky_notes):
                    target_note = sticky_notes[note_idx]
                    note_idx += 1
                else:
                    target_note = Square(side_length=0.25, fill_color=COLOR_MEAN, fill_opacity=1.0, stroke_width=1).scale(0.6)
                    final_notes.add(target_note)
                
                # Stack positioning starting from F2 column
                # Horizontal spacing ~0.5, Vertical spacing ~0.2
                target_pos = self.grid["F2"] + RIGHT * (col_idx * 0.5) + UP * (row_idx * 0.2)
                all_animations.append(target_note.animate.move_to(target_pos))

        self.play(*all_animations, FadeIn(final_notes), run_time=2)
        self.wait(2)
