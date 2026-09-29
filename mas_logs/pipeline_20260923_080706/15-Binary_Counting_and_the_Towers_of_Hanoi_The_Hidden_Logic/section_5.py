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
        lecture_lines = ["Binary counting defines the Hanoi solution.", 
                         "Recursive problems often share this structure.", 
                         "Simple binary logic hides deep complexity."]
        
        self.setup_layout("Conclusion: The Hidden Logic", lecture_lines)
        
        # Create lecture texts
        lecture_group = VGroup(*[Text(line, font_size=20, color=WHITE) for line in lecture_lines]).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(lecture_group, 'B1', 'C4', scale_factor=1.0)
        self.add(lecture_group)
        
        # Create key concepts
        key_concepts = VGroup(
            Text("Bits", color=BLUE),
            Text("Hanoi", color=GREEN),
            Text("Binary", color=YELLOW),
            Text("Complexity", color=RED)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(key_concepts, 'A3', scale_factor=0.8)
        
        # Create summary
        summary = Text("Logic is everywhere", font_size=40, color="#FFFF00")
        self.place_in_area(summary, 'D1', 'E6', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(Write(lecture_group[0]))
        self.play(lecture_group[0].animate.set_color("#FFFFFF"))
        self.play(FadeIn(key_concepts[0]), FadeIn(key_concepts[1]))
        
        # === Animation for Lecture Line 2 ===
        self.play(Write(lecture_group[1]))
        self.play(lecture_group[1].animate.set_color("#FFFF00"))
        self.play(FadeIn(key_concepts[2]))
        
        # === Animation for Lecture Line 3 ===
        self.play(Write(lecture_group[2]))
        self.play(lecture_group[2].animate.set_color("#FFFFFF"))
        self.play(FadeIn(key_concepts[3]))
        self.play(FadeIn(summary))
        
        self.wait(2)
