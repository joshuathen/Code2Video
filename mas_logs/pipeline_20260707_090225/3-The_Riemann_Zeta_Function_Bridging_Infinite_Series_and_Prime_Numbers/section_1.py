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
        # Helper function to generate the staircase shape
        def get_stair_path(heights, color):
            pts = [ORIGIN]
            curr = ORIGIN.copy()
            for h in heights:
                curr += UP * h
                pts.append(curr.copy())
                curr += RIGHT * 0.15 # Width of each step
                pts.append(curr.copy())
            return VMobject(color=color).set_points_as_corners(pts)

        # Initialize layout
        self.setup_layout(
            "The Infinite Staircase (Introduction)", 
            [
                'Meet Zeno, who climbs steps of decreasing height.', 
                'The harmonic series grows to infinity, step by step.', 
                'But summing inverse squares reaches a finite ceiling.'
            ]
        )

        # PREPARATION OF ASSETS
        # 1. Harmonic staircase (5 steps)
        h_5 = [1/(i+1) for i in range(5)]
        stair1 = get_stair_path(h_5, "#00FF00")
        
        # 2. Harmonic staircase (25 steps)
        h_25 = [1/(i+1) for i in range(25)]
        stair2 = get_stair_path(h_25, "#00FF00")
        
        # 3. Inverse squares staircase (25 steps)
        s_25 = [1/((i+1)**2) for i in range(25)]
        stair3 = get_stair_path(s_25, "#FFFF00")
        
        # 4. Zeno the Robot (White)
        zeno_head = Circle(radius=0.1, color=WHITE, fill_opacity=1)
        zeno_body = Square(side_length=0.2, color=WHITE, fill_opacity=1).next_to(zeno_head, DOWN, buff=0.02)
        zeno = VGroup(zeno_head, zeno_body)
        
        # 5. Infinity annotation (Red)
        inf_text = Text("Height -> ∞", color="#FF0000", font_size=20)
        
        # SCALING AND ALIGNMENT
        # We group staircases to ensure they share the same scale and bottom-left origin
        all_stairs = VGroup(stair1, stair2, stair3)
        # Fix Issue 38: Move from B1 to B2 to avoid crowding lecture notes
        self.place_in_area(all_stairs, 'B2', 'F6', scale_factor=0.8)
        
        # Re-align them to their common starting point in the grid after group placement
        origin_pos = stair2.get_start()
        stair1.move_to(origin_pos, aligned_edge=DL)
        stair3.move_to(origin_pos, aligned_edge=DL)
        
        # Place Zeno at the base
        zeno.move_to(origin_pos + LEFT*0.1 + UP*0.1)
        
        # Infinity text positioning
        # Fix Issue 39: Positioning fix for clutter
        self.place_in_area(inf_text, 'A5', 'A6', scale_factor=0.7)
        
        # Ceiling Line Preparation (Pink)
        # Sum of 1/n^2 for n=1 to infinity is pi^2/6 approx 1.6449
        # Scale math units to scene units using stair2 as reference. Use .height to avoid deprecation warning.
        v_scale = stair2.height / sum(h_25)
        ceiling_y = origin_pos[1] + (1.6449 * v_scale)
        
        # Shifted line start to match staircase position (Column 2)
        ceiling_line = DashedLine(
            start=[self.grid['B2'][0], ceiling_y, 0],
            end=[self.grid['B6'][0], ceiling_y, 0],
            color="#FF00FF"
        )
        
        # Replace MathTex with Text to avoid FileNotFoundError: 'latex' in environments without LaTeX installed.
        ceiling_label = Text("y = π²/6", font_size=24, color="#FF00FF")
        # Fix Issue 40: Position fix to avoid overlap
        self.place_in_area(ceiling_label, 'C5', 'C6', scale_factor=0.7)

        # ANIMATION SEQUENCE
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00") # Matching Green
        self.play(Create(stair1), FadeIn(zeno))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000") # Matching Red
        self.play(
            Transform(stair1, stair2),
            Write(inf_text),
            zeno.animate.move_to(stair2.get_end() + UP*0.1)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00") # Matching Yellow
        self.play(
            Transform(stair1, stair3),
            FadeOut(inf_text),
            Create(ceiling_line),
            Write(ceiling_label),
            zeno.animate.move_to(stair3.get_end() + UP*0.1)
        )
        self.wait(2)
