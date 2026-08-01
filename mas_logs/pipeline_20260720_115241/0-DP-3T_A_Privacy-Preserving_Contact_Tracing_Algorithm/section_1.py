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
        title = "Introduction: The Need for Contact Tracing"
        lecture_lines = [
            "Intro: Contact tracing is vital for disease control.",
            "Traditional methods are slow and intrusive.",
            "We need a digital, privacy-preserving solution.",
            "DP-3T offers a solution.",
            "It prioritizes user privacy."
        ]
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Animate the appearance of a virus spreading through a crowd of people.
        # Use a red virus spread animation. Position it in the center of the grid.
        virus = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/virus.svg", color=RED).scale(0.5)
        self.place_at_grid(virus, "C3")
        self.play(FadeIn(virus))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show icons representing people getting infected.
        # Use a grayscale person icon, and make it red when infected.
        people_group = VGroup()
        person_positions = ["B2", "B4", "D2", "D4", "E3"]
        for i, pos in enumerate(person_positions):
            person = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg", color=GRAY).scale(0.3)
            self.place_at_grid(person, pos)
            people_group.add(person)
        self.play(FadeIn(people_group))
        self.wait(0.5)

        infected_people_animations = []
        for person in people_group:
            infected_people_animations.append(person.animate.set_color(RED))
        self.play(AnimationGroup(*infected_people_animations, lag_ratio=0.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Illustrate the need for a system to track contacts.
        # Draw lines connecting infected people to others.
        # Use a gray line for tracing connections.
        connection_lines = VGroup()
        for i in range(len(people_group) - 1):
            for j in range(i + 1, len(people_group)):
                line = Line(people_group[i].get_center(), people_group[j].get_center(), color=GRAY, stroke_width=2)
                connection_lines.add(line)
        self.play(Create(connection_lines, lag_ratio=0.1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Animate the concept of a digital contact tracing system emerging.
        # Show a stylized digital interface/shield with a green checkmark.
        # Position it in the center of the grid.
        digital_system = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shield.svg", color=GREEN).scale(0.8)
        self.place_at_grid(digital_system, "C3")
        self.play(FadeIn(digital_system))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Conclude with a question: How can we trace contacts effectively and privately?
        # Display the question text.
        question = Text("How can we trace contacts effectively and privately?", font_size=20, color=WHITE)
        self.place_in_area(question, "E1", "F6")
        self.play(Write(question))
        self.wait(2)
