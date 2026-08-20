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
        self.setup_layout("Putting it Together: Strategy Implementation", 
                          ["Maximize letter distribution in move one.", 
                           "Feedback helps eliminate word candidates.", 
                           "Pick the next highest entropy word."])
        
        # === Animation for Lecture Line 1 ===
        # Draw a flowchart of strategy implementation steps [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/keyboard.svg] #FFFFFF.
        keyboard = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/keyboard.svg", color=WHITE)
        flowchart = VGroup(
            keyboard,
            RoundedRectangle(height=0.8, width=2.5, color=WHITE),
            Arrow(start=UP, end=DOWN, color=GRAY),
            RoundedRectangle(height=0.8, width=2.5, color=WHITE)
        ).arrange(DOWN)
        self.place_at_grid(flowchart, 'B5', scale_factor=0.6)
        self.play(FadeIn(flowchart))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Show state transition during decision making #00FFFF
        node1 = Circle(radius=0.3, color=WHITE)
        node2 = Circle(radius=0.3, color=WHITE)
        transition = Arrow(node1.get_right(), node2.get_left(), color=BLUE_A)
        state_group = VGroup(node1, transition, node2).arrange(RIGHT)
        self.place_at_grid(state_group, 'C5', scale_factor=0.8)
        self.play(FadeIn(state_group))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Display final strategy execution loop [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/monitor.svg] #00FF00.
        monitor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/monitor.svg", color=GREEN)
        loop_circle = Circle(radius=0.8, color=GREEN)
        loop_group = VGroup(monitor, loop_circle)
        self.place_at_grid(loop_group, 'D5', scale_factor=0.7)
        self.play(DrawBorderThenFill(loop_circle), FadeIn(monitor))
        self.lecture[2].set_color("#00FF00")
        
        self.wait(2)
