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
        self.setup_layout("Summary & Security Properties", [
            "Matching happens on devices locally.", 
            "Rotating IDs protect user anonymity.", 
            "No central contact database exists."
        ])
        
        # Asset: smartphone.svg
        phone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg")
        self.place_in_area(phone, 'B1', 'E6', scale_factor=1.5)
        self.add(phone)
        
        # === Animation for Lecture Line 1 ===
        # Display list of security properties
        prop1 = Text("Local Matching", font_size=20, color=WHITE)
        self.place_in_area(prop1, 'B1', 'B3', scale_factor=0.8)
        self.play(Write(prop1))
        self.lecture[0].set_color(BLUE)
        
        check1 = Checkmark(color=GREEN).next_to(prop1, RIGHT)
        self.play(FadeIn(check1))

        # === Animation for Lecture Line 2 ===
        # Rotating IDs
        prop2 = Text("Rotating IDs", font_size=20, color=WHITE)
        self.place_in_area(prop2, 'C1', 'C3', scale_factor=0.8)
        self.play(Write(prop2))
        self.lecture[1].set_color(BLUE)
        
        check2 = Checkmark(color=GREEN).next_to(prop2, RIGHT)
        self.play(FadeIn(check2))

        # === Animation for Lecture Line 3 ===
        # No Central DB
        prop3 = Text("No Central Database", font_size=20, color=WHITE)
        self.place_in_area(prop3, 'D1', 'D3', scale_factor=0.8)
        self.play(Write(prop3))
        self.lecture[2].set_color(BLUE)
        
        check3 = Checkmark(color=GREEN).next_to(prop3, RIGHT)
        self.play(FadeIn(check3))

        # Final Summary
        summary_box = Rectangle(color=YELLOW, height=1.5, width=3)
        summary_text = Text("Privacy Shield Active", font_size=20, color=YELLOW)
        summary = VGroup(summary_box, summary_text).arrange(DOWN)
        self.place_in_area(summary, 'E1', 'F4', scale_factor=0.9)
        self.play(Create(summary_box), Write(summary_text))
        self.wait(2)

class Checkmark(VMobject):
    def __init__(self, color=GREEN, **kwargs):
        super().__init__(**kwargs)
        self.add(Line(UP*0.1 + LEFT*0.1, DOWN*0.05 + LEFT*0.05).set_color(color))
        self.add(Line(DOWN*0.05 + LEFT*0.05, UP*0.15 + RIGHT*0.15).set_color(color))
        self.scale(0.8)
