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
            "Individual events are often unpredictable and chaotic.",
            "But groups of events follow strict, predictable patterns.",
            "This creates order from chaos, forming a bell curve."
        ]
        self.setup_layout("The Hook: Order from Chaos", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight Line 1 in Yellow
        self.lecture[0].set_color("#FFD700")
        
        # Show many yellow dots ('Space Rabbits') moving randomly
        num_dots = 45
        dots = VGroup(*[Dot(color="#FFD700", radius=0.07) for _ in range(num_dots)])
        
        # Issue 23 Fix: Use suggested area B1 to D6 for scattered dots
        # This call sets the initial scale and center for the dots group
        self.place_in_area(dots, 'B1', 'D6', scale_factor=0.8)
        
        # Bounds for B1 to D6 for internal distribution and jittering
        tl = self.grid["B1"]
        br = self.grid["D6"]
        
        # Randomize dot positions within the area B1-D6
        for dot in dots:
            dot.move_to([
                np.random.uniform(tl[0], br[0]),
                np.random.uniform(br[1], tl[1]),
                0
            ])
            
        self.add(dots)
        
        # Define jitter movement using an updater
        def jitter_dots(d, dt):
            # Constant small random shifts to simulate chaos
            d.shift(np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), 0]))
            # Constrain dots within the defined container B1 to D6
            if d.get_x() < tl[0]: d.set_x(tl[0])
            if d.get_x() > br[0]: d.set_x(br[0])
            if d.get_y() < br[1]: d.set_y(br[1])
            if d.get_y() > tl[1]: d.set_y(tl[1])

        for dot in dots:
            dot.add_updater(jitter_dots)
            
        self.wait(3)

        # === Animation for Lecture Line 2 ===
        # Reset line 1 color and highlight line 2 in white
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFFFF")
        
        # Issue 24 Fix: axis at Row F (centered in area F1-F6)
        axis = Line(LEFT * 2.5, RIGHT * 2.5, color=WHITE)
        self.place_in_area(axis, 'F1', 'F6')
        
        self.play(Create(axis))
        
        # Dots stop jittering and move to form stacks
        for dot in dots:
            dot.remove_updater(jitter_dots)
            
        # Using a binomial distribution (n=5, p=0.5) to simulate order from randomness
        bins = [[] for _ in range(6)]
        for dot in dots:
            bin_idx = np.random.binomial(5, 0.5)
            bins[bin_idx].append(dot)
            
        stack_anims = []
        for col_idx in range(6):
            col_key = str(col_idx + 1)
            # Use the grid position of the bottom row (F) as stack base for each column
            base_pos = self.grid[f"F{col_key}"]
            for stack_idx, dot in enumerate(bins[col_idx]):
                # Stack dots vertically on the axis at each grid column
                target = base_pos + UP * (0.15 + stack_idx * 0.16)
                stack_anims.append(dot.animate.move_to(target))
                
        self.play(*stack_anims, run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reset line 2 and highlight line 3 in Cyan
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        
        # A cyan bell curve outline fades in
        # Parameters chosen to fit the bin distribution (centered at col 3-4)
        center_x = (self.grid["F3"][0] + self.grid["F4"][0]) / 2
        baseline_y = self.grid["F1"][1]
        
        bell_curve = FunctionGraph(
            lambda x: 2.2 * np.exp(-((x - center_x)**2) / 1.5) + baseline_y,
            x_range=[self.grid["F1"][0]-0.5, self.grid["F6"][0]+0.5],
            color="#00FFFF"
        )
        
        # Issue 25 Fix: Group bell curve, dots, and axis, and place in B1-F6
        bell_curve_group = VGroup(axis, dots, bell_curve)
        
        # Final layout adjustment to utilize grid height and avoid compression
        target_center_pos = (self.grid["B1"] + self.grid["F6"]) / 2
        
        self.play(FadeIn(bell_curve))
        self.play(
            bell_curve_group.animate.scale(0.9).move_to(target_center_pos),
            run_time=2
        )
        
        self.wait(4)
