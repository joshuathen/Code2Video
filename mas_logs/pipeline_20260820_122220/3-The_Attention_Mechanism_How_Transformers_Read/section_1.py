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
        self.setup_layout("The Intuitive Hook: The Cocktail Party Effect", 
                          ["Transformers use selective focus like a party.", 
                           "Humans filter noise to hear one speaker.", 
                           "Models filter relevant tokens from a sequence."])

        # Assets
        person_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg"
        ear_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ear.svg"
        
        # Animation Setup
        # Create representation of people talking
        people = VGroup(*[SVGMobject(person_asset, color="#888888") for _ in range(6)])
        
        # Grid nodes area
        self.place_in_area(people, 'B2', 'E5', scale_factor=0.8)
        
        # Focus person
        focus_person = SVGMobject(person_asset, color="#FFFF00")
        self.place_at_grid(focus_person, 'C3', scale_factor=1.0)
        
        # Ear symbol
        ear = SVGMobject(ear_asset, color="#FFFFFF").next_to(focus_person, RIGHT, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(people), FadeIn(focus_person))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.play(FadeIn(ear))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Fade out surrounding voices
        self.play(
            people.animate.set_opacity(0.2),
            focus_person.animate.set_color(YELLOW),
            run_time=2
        )
        self.wait(2)
