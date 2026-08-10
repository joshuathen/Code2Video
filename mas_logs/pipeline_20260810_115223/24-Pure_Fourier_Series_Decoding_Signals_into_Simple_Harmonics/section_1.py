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
        self.setup_layout("Prerequisites & Intuition", [
            "Periodic signals repeat their behavior over time.", 
            "Like coordinates, sine waves act as basic building blocks.", 
            "A musical chord is a sum of simple tones."
        ])
        
        # Animation Elements
        # Assets: 
        # speaker.svg, piano.svg, guitar.svg
        icon_speaker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg", color="#00FF00")
        label_speaker = Text("Periodic", font_size=16, color="#00FF00").next_to(icon_speaker, DOWN)
        group_speaker = VGroup(icon_speaker, label_speaker)
        
        icon_piano = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/piano.svg", color="#FFFFFF")
        label_piano = Text("Sine", font_size=16, color="#FFFFFF").next_to(icon_piano, DOWN)
        group_piano = VGroup(icon_piano, label_piano)
        
        icon_guitar = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/guitar.svg", color="#FFD700")
        label_guitar = Text("Spectrum", font_size=16, color="#FFD700").next_to(icon_guitar, DOWN)
        group_guitar = VGroup(icon_guitar, label_guitar)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.place_at_grid(group_speaker, 'B2', scale_factor=0.6)
        self.play(FadeIn(group_speaker))
        self.play(Indicate(group_speaker, color="#00FF00"))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.place_at_grid(group_piano, 'B5', scale_factor=0.6)
        self.play(FadeIn(group_piano))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.place_at_grid(group_guitar, 'E3', scale_factor=0.6)
        self.play(FadeIn(group_guitar))
        self.wait(1)
