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
        # Setup the layout with specific lines
        lines = [
            "Digital contact tracing helps stop virus spread quickly.",
            "But centralized databases risk mass government surveillance.",
            "Can we trace contacts without knowing locations or identities?"
        ]
        self.setup_layout("The Tracing Dilemma", lines)
        
        # Internal helper for person icons
        def create_person_icon(color=WHITE):
            head = Circle(radius=0.15, color=color, fill_opacity=1)
            # Body as a trapezoid
            body = Polygon(
                [-0.2, 0, 0], [0.2, 0, 0], [0.3, -0.4, 0], [-0.3, -0.4, 0], 
                color=color, fill_opacity=1
            ).next_to(head, DOWN, buff=0.05)
            return VGroup(head, body)

        # === Animation for Lecture Line 1 ===
        # Line 1 Highlight: Yellow. Virus spread: Red elements.
        self.lecture[0].set_color(YELLOW)
        
        people_group = VGroup()
        grid_positions = ["B2", "B4", "B6", "D2", "D4", "D6", "F2", "F4", "F6"]
        for pos in grid_positions:
            p = create_person_icon(BLUE_A)
            self.place_at_grid(p, pos, scale_factor=0.6)
            people_group.add(p)
            
        self.play(Create(people_group))
        self.wait(0.5)
        
        # Virus spread animation
        infected_indices = [1, 3, 4, 8]
        virus_color = "#e74c3c"
        self.play(*[people_group[i].animate.set_color(virus_color) for i in infected_indices], run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE) # Revert prev
        self.lecture[1].set_color("#e74c3c") # Red for risk
        
        # Transition to a central 'Big Brother' server icon [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/server.svg]
        server = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/server.svg")
        server.set_color(GRAY_B)
        self.place_at_grid(server, "D4", scale_factor=0.8)
        
        # Surveillance label - Issue 38, 40: Positioned at A4 to avoid overlap and act as header
        surv_label = Text("Surveillance Risk", color="#e74c3c", font_size=20, weight=BOLD)
        self.place_at_grid(surv_label, "A4", scale_factor=1.0)

        # Centralized connections
        connections = VGroup()
        for i, p in enumerate(people_group):
            if i == 4: continue # Skip person at D4 as it's replaced by server
            l = Line(p.get_center(), server.get_center(), color="#e74c3c", stroke_width=1.5, stroke_opacity=0.6)
            connections.add(l)

        self.play(
            FadeIn(server),
            FadeIn(surv_label),
            Create(connections),
            # Icons turn grey to indicate being monitored
            people_group.animate.set_color(GRAY).scale(0.8),
            people_group[4].animate.set_opacity(0)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE) # Revert prev
        self.lecture[2].set_color("#2ecc71") # Green for privacy
        
        # Reset scene for Alice and Bob
        alice = create_person_icon(WHITE)
        bob = create_person_icon(WHITE)
        self.place_at_grid(alice, "D2", scale_factor=0.8)
        self.place_at_grid(bob, "D6", scale_factor=0.8)
        
        a_label = Text("Alice", font_size=18).next_to(alice, UP, buff=0.1)
        b_label = Text("Bob", font_size=18).next_to(bob, UP, buff=0.1)
        
        # Encounter distance line
        encounter_line = DoubleArrow(alice.get_right(), bob.get_left(), color=YELLOW, buff=0.1)
        
        # Privacy shield icon - Issue 39: Increased scale to 1.2
        shield_pts = [[-0.4, 0.4, 0], [0.4, 0.4, 0], [0.4, -0.1, 0], [0, -0.5, 0], [-0.4, -0.1, 0]]
        shield = Polygon(*shield_pts, color="#2ecc71", fill_opacity=1)
        self.place_at_grid(shield, "D4", scale_factor=1.2)
        shield_text = Text("Privacy\nFirst", font_size=14, color=WHITE).move_to(shield.get_center())

        self.play(
            FadeOut(server), FadeOut(connections), FadeOut(surv_label), FadeOut(people_group),
            FadeIn(alice), FadeIn(bob), FadeIn(a_label), FadeIn(b_label)
        )
        self.play(Create(encounter_line))
        self.play(DrawBorderThenFill(shield), Write(shield_text))
        self.wait(3)
