from manim import *
import os

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
        lecture_lines = ["Independence ignores external information relevance.", "Bayes' Theorem uses information to transform.", "Both frame how we perceive uncertainty."]
        self.setup_layout("Synthesis & Summary", lecture_lines)
        
        # Animations
        # Colors: light, distinguishable
        color1 = "#4FC3F7"  # Sky Blue
        color2 = "#FFD54F"  # Amber
        color3 = "#81C784"  # Light Green
        
        # === Animation for Lecture Line 1 ===
        # Summarize Bayes' theorem steps as a flowchart
        node1 = Circle(radius=0.3, color=color1, fill_opacity=0.5)
        node1_label = Text("P(A|B)", font_size=16)
        flow1 = VGroup(node1, node1_label)
        self.place_at_grid(flow1, 'C2', scale_factor=0.8)
        self.play(FadeIn(flow1))
        self.lecture[0].set_color(color1)

        # === Animation for Lecture Line 2 ===
        # Recap: Conditional probability, independence, and Bayes' formula
        node2 = Circle(radius=0.3, color=color2, fill_opacity=0.5)
        node2_label = Text("Bayes", font_size=16)
        flow2 = VGroup(node2, node2_label)
        self.place_at_grid(flow2, 'C4', scale_factor=0.8)
        line = Line(flow1.get_right(), flow2.get_left())
        self.play(FadeIn(flow2), Create(line))
        self.lecture[1].set_color(color2)

        # === Animation for Lecture Line 3 ===
        # Final view of the robot navigation success scenario
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        robot = Circle(radius=0.3, color=color3, fill_opacity=0.5) # Fallback if svg is not visual
        self.place_in_area(robot, 'D4', 'F6', scale_factor=0.6)
        self.play(FadeIn(robot))
        self.lecture[2].set_color(color3)
        self.wait(2)
