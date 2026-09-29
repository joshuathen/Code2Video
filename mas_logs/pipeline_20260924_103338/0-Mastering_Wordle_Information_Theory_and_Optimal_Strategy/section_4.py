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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application: The Robot vs. The Human", ["Humans guess intuitively.", "Robots calculate information density.", "Robots solve in 3.5 guesses."])
        
        # === Animation for Lecture Line 1 ===
        # Using SVG Assets
        robot_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", color="#00FF00")
        human_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/human.svg", color="#FF0000")
        
        robot_label = Text("Robot", color="#00FF00", font_size=24)
        human_label = Text("Human", color="#FF0000", font_size=24)
        
        self.place_at_grid(robot_label, 'B2')
        self.place_at_grid(human_label, 'B5')
        self.play(FadeIn(robot_label), FadeIn(human_label))
        self.lecture[0].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Re-using the SVG assets as icons
        self.place_at_grid(robot_svg, 'A3', scale_factor=0.6)
        self.place_at_grid(human_svg, 'A4', scale_factor=0.6)
        
        self.play(FadeIn(robot_svg), FadeIn(human_svg))
        
        # Animate Robot moving towards center
        self.play(robot_svg.animate.move_to(self.grid['C3']))
        
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        score_board = Rectangle(height=1.5, width=3, color=WHITE)
        self.place_in_area(score_board, 'E3', 'F5', scale_factor=0.6)
        
        score_text = Text("Final Score: Robot 3.5 | Human 5+", font_size=20, color="#FFFF00")
        self.place_in_area(score_text, 'E3', 'E5', scale_factor=0.5)
        
        self.play(Create(score_board), Write(score_text))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
