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
        self.setup_layout("Summary & Quick Check", [
            "Implicit differentiation uses the Chain Rule.",
            "Derivatives contain both x and y.",
            "Always verify your steps carefully."
        ])
        
        # NOTE: Using placeholder for icons as per asset path in storyboard
        # Since '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg' is literally specified, using standard Mobjects as fallback
        # as asset/none.svg doesn't exist
        dot_explicit = Dot(color=WHITE)
        robot_arm = VGroup(
            Line(ORIGIN, UP*1, color=WHITE),
            Line(UP*1, UP*1 + RIGHT*1, color=WHITE)
        )
        square = Square(side_length=0.5, color=WHITE)

        # Positioning as requested
        self.place_at_grid(dot_explicit, 'B5', scale_factor=0.5)
        self.place_in_area(robot_arm, 'E2', 'F4', scale_factor=0.8)
        self.place_at_grid(square, 'C2', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color('#FFFFFF'))
        self.play(FadeIn(dot_explicit))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color('#FF00FF'))
        self.play(Create(robot_arm))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color('#00FFFF'))
        self.play(FadeIn(square))
        self.wait(2)
