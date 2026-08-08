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

class Section2Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        self.setup_layout("Prerequisite: The Anatomy of a Vector", 
                          ["A vector is a recipe of basis vectors.", 
                           "Vector [3, 2] means 3 i-steps and 2 j-steps.", 
                           "We reach the target by scaling and adding arrows."])
        
        # Global Colors
        color_i = "#0000FF" # blue
        color_j = "#00FF00" # green
        color_result = "#800080" # purple
        color_highlight = YELLOW

        # Origin for our vector space (centered on grid)
        origin_pos = self.grid['D2']

        # === Animation for Lecture Line 1 ===
        # Draw standard unit vectors i (blue) and j (green) at the origin.
        
        # i vector: D2 -> D3 (1 step)
        i_vec = Arrow(start=origin_pos, end=self.grid['D3'], buff=0, color=color_i)
        i_label = MathTex("\\hat{i}", color=color_i, font_size=32)
        self.place_at_grid(i_label, 'E3', scale_factor=0.7) # Positioned below tip, scaled per Issue 40
        
        # j vector: D2 -> C2 (1 step up)
        j_vec = Arrow(start=origin_pos, end=self.grid['C2'], buff=0, color=color_j)
        j_label = MathTex("\\hat{j}", color=color_j, font_size=32)
        self.place_at_grid(j_label, 'C1', scale_factor=0.7) # Positioned left of tip, scaled per Issue 41
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            Create(i_vec), 
            Create(j_vec), 
            Write(i_label), 
            Write(j_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the robot moving 3 units along i, then 2 units along j.
        self.play(self.lecture[1].animate.set_color(color_highlight))
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        robot.set_color(WHITE)
        robot.scale(0.3)
        robot.move_to(origin_pos)
        self.add(robot)
        
        # Path sequence:
        # Move along x: D2 -> D3 -> D4 -> D5 (3 steps)
        # Move along y: D5 -> C5 -> B5 (2 steps)
        move_targets = ['D3', 'D4', 'D5', 'C5', 'B5']
        
        for target in move_targets:
            self.play(robot.animate.move_to(self.grid[target]), run_time=0.6, rate_func=linear)
            
        # Draw component vectors (scaled basis vectors)
        # i_scaled: 3 units
        i_scaled = Arrow(start=origin_pos, end=self.grid['D5'], buff=0, color=color_i, stroke_width=2)
        # j_scaled: 2 units
        j_scaled = Arrow(start=self.grid['D5'], end=self.grid['B5'], buff=0, color=color_j, stroke_width=2)
        
        self.play(Create(i_scaled), Create(j_scaled))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw the resulting vector [3, 2] from origin to the robot's position in purple.
        self.play(self.lecture[2].animate.set_color(color_result))
        
        res_vec = Arrow(start=origin_pos, end=self.grid['B5'], buff=0, color=color_result, stroke_width=6)
        res_label = MathTex("\\vec{v} = \\begin{bmatrix} 3 \\\\ 2 \\end{bmatrix}", color=color_result, font_size=32)
        
        # Issue 39: Positioning res_label in area A5-C6 to avoid clipping
        self.place_in_area(res_label, 'A5', 'C6', scale_factor=0.8)
        
        self.play(Create(res_vec), Write(res_label))
        self.wait(3)
