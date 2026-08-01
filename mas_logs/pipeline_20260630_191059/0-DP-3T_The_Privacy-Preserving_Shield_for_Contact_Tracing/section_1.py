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
        title_text = "The Privacy Dilemma: Public Health vs. Personal Secret"
        lecture_lines = [
            "Traditional contact tracing often risks personal privacy.",
            "Centralized servers track your location and identity.",
            "DP-3T offers a privacy-first, decentralized alternative."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define Colors
        VIRUS_COLOR = "#FF4500" # OrangeRed
        EYE_COLOR = "#FFFFFF" # White
        SHIELD_COLOR = "#00FF7F" # SpringGreen
        NODE_COLOR = WHITE
        
        # === Animation for Lecture Line 1 ===
        # Color current lecture line
        self.lecture[0].set_color(VIRUS_COLOR)
        
        # Create user network
        nodes = VGroup(*[Circle(radius=0.1, color=NODE_COLOR, fill_opacity=1, stroke_width=2) for _ in range(6)])
        # Fixing Issue 29: Move node from B2 to B3 to accommodate virus shift
        node_grid_positions = ["B3", "B5", "C4", "D2", "E5", "F3"]
        for i, pos in enumerate(node_grid_positions):
            self.place_at_grid(nodes[i], pos)
            
        edges = VGroup(
            Line(nodes[0].get_center(), nodes[2].get_center(), color=GREY, stroke_width=2),
            Line(nodes[1].get_center(), nodes[2].get_center(), color=GREY, stroke_width=2),
            Line(nodes[2].get_center(), nodes[3].get_center(), color=GREY, stroke_width=2),
            Line(nodes[2].get_center(), nodes[4].get_center(), color=GREY, stroke_width=2),
            Line(nodes[3].get_center(), nodes[5].get_center(), color=GREY, stroke_width=2),
        )
        
        # Virus Mobject (represented as a spiked circle)
        virus_core = Circle(radius=0.15, color=VIRUS_COLOR, fill_opacity=1)
        spikes = VGroup(*[Line(ORIGIN, 0.25 * RIGHT, color=VIRUS_COLOR, stroke_width=3).rotate(a) for a in np.linspace(0, 2*PI, 8)])
        virus = VGroup(virus_core, spikes)
        # Fixing Issue 29: Position virus at B3 with scale 0.8
        self.place_at_grid(virus, "B3", scale_factor=0.8)
        
        self.play(Create(nodes), Create(edges), run_time=1.5)
        self.play(FadeIn(virus))
        
        # Virus spreading sequence
        self.play(nodes[0].animate.set_color(VIRUS_COLOR))
        self.play(virus.animate.move_to(self.grid["C4"]), run_time=0.8)
        self.play(nodes[2].animate.set_color(VIRUS_COLOR))
        self.play(virus.animate.move_to(self.grid["D2"]), run_time=0.8)
        self.play(nodes[3].animate.set_color(VIRUS_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update lecture line colors
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(EYE_COLOR)
        )
        
        # Create 'Eye' icon
        start_pt = np.array([-0.5, 0, 0])
        end_pt = np.array([0.5, 0, 0])
        eye_top = ArcBetweenPoints(start_pt, end_pt, angle=-PI/2, color=EYE_COLOR)
        eye_bottom = ArcBetweenPoints(start_pt, end_pt, angle=PI/2, color=EYE_COLOR)
        iris = Circle(radius=0.15, color=EYE_COLOR, fill_opacity=0.5)
        pupil = Dot(radius=0.07, color=EYE_COLOR)
        eye = VGroup(eye_top, eye_bottom, iris, pupil)
        # Fixing Issue 30: Scale eye to 1.0 in A3-A4
        self.place_in_area(eye, "A3", "A4", scale_factor=1.0)
        
        # Surveillance lines (data collection)
        data_lines = VGroup(*[
            Line(node.get_center(), eye.get_center(), color=EYE_COLOR, stroke_width=1).set_opacity(0.4)
            for node in nodes
        ])
        
        self.play(FadeIn(eye))
        self.play(Create(data_lines), run_time=2)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Update lecture line colors
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(SHIELD_COLOR)
        )
        
        # Shield icon
        shield_shape = Polygon(
            [-0.4, 0.4, 0], [0.4, 0.4, 0], [0.4, -0.2, 0], [0, -0.5, 0], [-0.4, -0.2, 0],
            color=SHIELD_COLOR, fill_opacity=0.8
        )
        shield = VGroup(shield_shape)
        # Fixing Issue 28: Move shield to A5-A6, scale 0.9
        self.place_in_area(shield, "A5", "A6", scale_factor=0.9)
        
        # Transition: Shield replaces Eye and data lines vanish
        self.play(
            ReplacementTransform(eye, shield),
            data_lines.animate.set_opacity(0),
        )
        
        # Protection effect (nodes glow or get a green circle)
        protections = VGroup(*[
            Circle(radius=0.18, color=SHIELD_COLOR, stroke_width=3).move_to(node.get_center())
            for node in nodes
        ])
        
        self.play(Create(protections), run_time=1.5)
        self.wait(3)
