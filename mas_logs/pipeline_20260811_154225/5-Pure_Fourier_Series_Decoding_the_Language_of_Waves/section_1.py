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
        lecture_lines = ["Complex periodic signals are just chords of sine waves.", "A cat’s purr reveals a base and harmonics.", "Chaotic waves resolve into ordered stacks of harmonics."]
        self.setup_layout("The Hook: The Musical Analogy", lecture_lines)
        
        # Define animated objects
        staff = VGroup(*[Line(LEFT*1.5, RIGHT*1.5) for _ in range(5)]).arrange(DOWN, buff=0.2)
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        
        melody_line = VGroup(*[Dot(color=YELLOW) for _ in range(5)])
        
        # === Animation for Lecture Line 1 ===
        # Display a musical staff with scattered notes appearing alongside a cat.png. (Color: #FFFFFF)
        self.place_at_grid(staff, 'D4', scale_factor=0.6)
        self.place_at_grid(cat_icon, 'B4', scale_factor=0.5)
        
        self.play(FadeIn(staff), FadeIn(cat_icon))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Animate notes coalescing into a single clear melody line. (Color: #FFFF00)
        # Note: Cat remains present to illustrate 'purr'
        self.play(FadeIn(melody_line))
        for i, dot in enumerate(melody_line):
            self.place_at_grid(dot, f"C{i+2}")
            
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Highlight the melody line with a vibrant pulse featuring a cat.png. (Color: #00FFFF)
        self.play(Indicate(melody_line, color="#00FFFF"), cat_icon.animate.set_color("#00FFFF"))
        self.lecture[2].set_color("#00FFFF")
        self.wait(1)
