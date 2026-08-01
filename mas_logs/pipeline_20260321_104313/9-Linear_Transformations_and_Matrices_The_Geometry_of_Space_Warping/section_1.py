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
        # Section title and lecture lines as per mandatory structure
        title_text = "Prerequisites: Vectors as Directions"
        lecture_lines = [
            "Our robot starts at coordinate one, one.",
            "The grid is built on unit vectors i-hat and j-hat.",
            "Moving two steps along i-hat shifts its horizontal position.",
            "Adding one step along j-hat reaches the target point.",
            "This movement creates the final vector, three comma two."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Create a grid using the provided grid points A1-F6
        grid_elements = VGroup()
        for i in range(6):  # Rows A-F
            row_char = chr(65 + i)
            line = Line(self.grid[f"{row_char}1"], self.grid[f"{row_char}6"], color="#444444", stroke_width=1)
            grid_elements.add(line)
        for j in range(1, 7):  # Cols 1-6
            line = Line(self.grid[f"A{j}"], self.grid[f"F{j}"], color="#444444", stroke_width=1)
            grid_elements.add(line)
            
        # Origin point for reference (D3)
        origin_dot = Dot(self.grid['D3'], radius=0.05, color=GRAY)
        
        # Robot (yellow dot) starting at (1,1) relative to origin D3 -> C4
        robot = Dot(color="#FFFF00")
        self.place_at_grid(robot, 'C4')
        robot_label = Text("Robot", font_size=18, color="#FFFF00")
        # Issue 32 fix: scale_factor=0.7 to avoid clipping
        self.place_at_grid(robot_label, 'B4', scale_factor=0.7)
        
        self.play(Create(grid_elements), FadeIn(origin_dot))
        self.play(FadeIn(robot), Write(robot_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        
        # i-hat (Red) and j-hat (Green) from origin D3
        i_hat = Arrow(self.grid['D3'], self.grid['D4'], color="#FF0000", buff=0, stroke_width=4)
        j_hat = Arrow(self.grid['D3'], self.grid['C3'], color="#00FF00", buff=0, stroke_width=4)
        
        i_label = Text("i", color="#FF0000", font_size=24, slant=ITALIC)
        j_label = Text("j", color="#00FF00", font_size=24, slant=ITALIC)
        
        # Positioning labels within 1 grid unit
        self.place_at_grid(i_label, 'E4') # Below i-hat tip
        # Issue 33 fix: Moved to B3 with scale_factor=0.8 for symmetry
        self.place_at_grid(j_label, 'B3', scale_factor=0.8)
        
        self.play(GrowArrow(i_hat), Write(i_label))
        self.play(GrowArrow(j_hat), Write(j_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Fade out the initial robot label to keep movement clear
        self.play(FadeOut(robot_label))
        
        # Move 2 steps along i-hat: (1,1) to (3,1) -> C4 to C6
        self.play(robot.animate.move_to(self.grid['C6']), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Highlight lecture line 4
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        
        # Move 1 step along j-hat: (3,1) to (3,2) -> C6 to B6
        self.play(robot.animate.move_to(self.grid['B6']), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Highlight lecture line 5
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        # Final Vector from origin D3 to the robot's end position B6
        final_vector = Arrow(self.grid['D3'], self.grid['B6'], color="#FFFFFF", buff=0, stroke_width=5)
        
        vector_coords = Text("[3, 2]", color="#FFFFFF", font_size=28)
        # Issue 31 fix: Positioned at A5 with scale_factor=0.7 to prevent clipping
        self.place_at_grid(vector_coords, 'A5', scale_factor=0.7)
        
        self.play(GrowArrow(final_vector))
        self.play(Write(vector_coords))
        self.wait(3)
